# dynamic.py
from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Annotated
from datetime import date, datetime
import asyncio
import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import json
from google.cloud import bigquery, storage
from google.oauth2 import service_account

from fastapi.responses import HTMLResponse
import pandas as pd
import os

key_path = "cloudkarya-internship-1c013aa63f5f.json"
bigquery_client = bigquery.Client.from_service_account_json(key_path)
storage_client = storage.Client.from_service_account_json(key_path)

project_id = "cloudkarya-internship"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageData(BaseModel):
    img_file: str
    img_type: str
    patient_id: str
    patient_fname: str
    patient_lname: str
    patient_dob: date
    patient_gender: str
    patient_email: str
    pneumonia_prob: float
    tuberculosis_prob: float
    cancer_prob: float
    covid19_prob: float


templates = Jinja2Templates(directory="templates")

@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.get("/PatientForm")
async def Patient_form(request: Request):
    return templates.TemplateResponse("PatientForm.html", {"request": request})

@app.get("/report")
async def report_fun(request: Request):
    return templates.TemplateResponse("report.html", {"request": request})



class FileRequest(BaseModel):
    file_path: str

@app.post("/upload")
async def report_file(request: Request,image:Annotated[UploadFile, File(...)],
                       patient_fname: Annotated[str,Form(...)],
                       patient_lname: Annotated[str,Form(...)],
                       patient_dob: Annotated[str,Form(...)],
                       patient_email: Annotated[str,Form(...)],
                       Gender: Annotated[str,Form(...)],
                       image_type:Annotated[str,Form(...)],
                       patient_mobile:Annotated[str,Form(...)],
                       patient_id:Annotated[str,Form(...)]
                       ):

    contents = await image.read()

    encoded_img=base64.b64encode(contents).decode('utf-8')
   
    # Create a client instance

    # Retrieve the bucket
    if image_type == 'X-ray':
        bucket_name = 'lung_abn_raw'
        folder_name = 'X-ray/'
    else:
        bucket_name = 'lung_abn_raw'
        folder_name = 'CT-scan/'
    bucket = storage_client.get_bucket(bucket_name)


# Format the date as "Month dd, YYYY"
    date_test=datetime.now()
    date_of_test = date_test.date()
    date_object = datetime.strptime(str(date_of_test), "%Y-%m-%d")
    date= date_object.strftime("%B %d, %Y")

    patient_name = patient_fname + " " + patient_lname 
    filename = f"{patient_id}"
    #blob.upload_from_file(image.file, content_type=image.content_type)
    blob = bucket.blob(f"{folder_name}/{filename}")
    
    image_path = f'https://storage.googleapis.com/{bucket_name}/{folder_name}/{filename}'
    
    image.file.seek(0)
    blob.upload_from_file(image.file, content_type=image.content_type)
    image.close()
    pneumonia_prob = 0.0
    tuberculosis_prob = 0.0
    cancer_prob = 0.0
    covid19_prob = 0.0
    query =  f"""
    INSERT INTO `{project_id}.ImageData2.ImageDataTable` (
        img_file, img_type, patient_id, patient_fname,patient_lname, patient_dob, patient_gender,
        patient_email,patient_phno, date_of_test,
        pneumonia_prob, tuberculosis_prob, cancer_prob, covid19_prob
    )
    VALUES (
        '{image_path}', '{image_type}', '{patient_id}', '{patient_fname}', '{patient_lname}',
        DATE('{patient_dob}'), '{Gender}', '{patient_email}', '{patient_mobile}',
        DATE('{date_of_test}'),
        {pneumonia_prob}, {tuberculosis_prob}, {cancer_prob}, {covid19_prob}
    )
    """
    job = bigquery_client.query(query)
    job.result()
    date_test=datetime.now()
    date_of_test = date_test.date()
    print(date_of_test)
    query = f"""
        SELECT pneumonia_prob, tuberculosis_prob, cancer_prob, covid19_prob
        FROM `{project_id}.ImageData2.ImageDataTable`
        WHERE patient_id = '{patient_id}'
    """

    query_job = bigquery_client.query(query)
    results = query_job.result()
    
    # Assign column values to variables
    for row in results:
        pred1 = row['pneumonia_prob']
        pred2 = row['tuberculosis_prob']
        pred3 = row['cancer_prob']
        pred4 = row['covid19_prob']
        print(pred1,pred2,pred3,pred4)

    await asyncio.sleep(60)
    return templates.TemplateResponse("report.html", {"request": request, "result1": pred1, "result2": pred2,
                                                     "result3": pred3, "result4": pred4, "img1":encoded_img,
                                                     "patient_name": patient_name, "patient_dob": patient_dob,
                                                     "patient_email": patient_email, "Gender": Gender,
                                                     "Uploaded_image": image_type,"date":str(date)})


@app.post("/ImageData/")
async def create_image_data(item: ImageData):
    query = f"""
    INSERT INTO `{project_id}.ImageData2.ImageDataTable`
    VALUES ('{item.img_file}', '{item.img_type}', '{item.patient_id}', '{item.patient_name}', 
            DATE('{item.patient_dob}'), '{item.patient_gender}', '{item.patient_email}', 
            {item.pneumonia_prob}, {item.tuberculosis_prob}, {item.cancer_prob}, {item.covid19_prob})
    """
    job = bigquery_client.query(query)
     # Wait for the query to complete

    return {"message": "Data inserted successfully"}


@app.get("/ImageDatas",response_class=HTMLResponse)
async def get_image_data():
   query = f"""
         SELECT  * FROM {project_id}.ImageData2.ImageDataTable;
   """
   df = bigquery_client.query(query).to_dataframe()
   # df.head()
   return df.to_html()


@app.get("/ImageData/{id}",response_class=HTMLResponse)
async def get_image_data(id):
   query = f"""
         SELECT  * FROM {project_id}.ImageData2.ImageDataTable
         WHERE patient_id = '{id}';
   """
   df = bigquery_client.query(query).to_dataframe()
   # df.head()
   return df.to_html()

@app.post("/getdata")
async def get_data(request: Request,patient_id:Annotated[str,Form(...)]):
   query = f"""
         SELECT  * FROM {project_id}.ImageData2.ImageDataTable
         WHERE patient_id ='{patient_id}';
   """
   df = bigquery_client.query(query).to_dataframe()
   print(df.head())
   image_path=df.iloc[0]['img_file']
   predi1=df.iloc[0]['pneumonia_prob']
   predi2=df.iloc[0]['tuberculosis_prob']
   predi4=df.iloc[0]['covid19_prob']
   predi3=df.iloc[0]['cancer_prob']
   patient_fname=df.iloc[0]['patient_fname']
   patient_lname=df.iloc[0]['patient_lname']
   patient_email=df.iloc[0]['patient_email']
   patient_dob=df.iloc[0]['patient_dob']
   Gender=df.iloc[0]['patient_gender']
   image_type=df.iloc[0]['img_type']
   date_of_test=df.iloc[0]['date_of_test']
   date_object = datetime.strptime(str(date_of_test), "%Y-%m-%d")
   date= date_object.strftime("%B %d, %Y")

   pred1=round(predi1*100,2)
   pred2=round(predi2*100,2)
   pred3=round(predi3*100,2)
   pred4=round(predi4*100,2)
   patient_name=patient_fname + " " + patient_lname 
   

 

   return templates.TemplateResponse("report.html", {"request": request, "result1":pred1,"result2":pred2,"result3":pred3, "result4":pred4, "img":image_path, "patient_name":patient_name,"patient_dob":patient_dob,"patient_email":patient_email,"Gender":Gender,"Uploaded_image":image_type,"date":date})
   
   # df.head()
   #    return df.to_html()   
   # 
    
@app.get("display/image")
async def display_image():
    bucket_name = "lung_abn"
    filename = "Lung_Images/5.png"

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    
    # Download the image from the bucket
    image_data = blob.download_as_bytes()

    # Return an HTML response with the image data embedded
    return templates.TemplateResponse("report.html", {"request": request, "image_content": f'<img src="data:image/jpeg;base64,{image_data.decode()}" alt="Image">'})
    # return HTMLResponse(content=f'<img src="data:image/jpeg;base64,{image_data.decode()}" alt="Image">')               
                
        







# # if __name__ == '__dynamic__':
# #    uvicorn.run(app, host='0.0.0.0', port=8000)

# demo = Image.open("/workspace/Pheonix_Squadron/PDR-OS-LRG.jpg")
# plt.imshow(demo)

"""@app.post("/images")
async def create_image(image: UploadFile = File(...)):
    db = SessionLocal()
    image_data = image.file.read()

    image_obj = schemas.ImageCreate(
        filename=image.filename,
        filepath=image.filepath,
        image_data=image_data
    )

    db_image = models.Image(**image_obj.dict())
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    return {"message": "Image stored in the database."}"""


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     img = Image.open(io.BytesIO(contents))
#     img = img.resize((150, 150))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)

#     model = tf.keras.models.load_model("/workspace/X-ray/pnuemonia_sequential1.h5")
#     predictions = model.predict(img)
#     predictions1 = predictions * 100
#     threshold = 0.5
#     binary_outputs = (predictions1 > threshold).astype(int)

#     result = {
#         "img": img,
#         "prediction": predictions1[0][0]
#     }

#     return templates.TemplateResponse("report.html", {"request": file, "result": result})