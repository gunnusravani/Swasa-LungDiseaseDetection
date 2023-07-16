from sqlalchemy import Column, Integer, String, LargeBinary  # Add LargeBinary import

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    img_file = Column(String, index=True)
    patient_name = Column(String)
    patient_dob = Column(Integer) 
    patient_email = Column(String)
    pnuemonia_prob=Column(Float)
    Tuberculosis_prob=Column(Float)
    Cancer_prob=Column(Float)
    Covid19_prob=Column(Float)


