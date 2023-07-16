// script.js

// Example function for handling the image upload form submission
function handleImageUpload(event) {
    event.preventDefault();
  
    // Get the file input element and the selected file
    const fileInput = document.querySelector('input[type="file"]');
    const file = fileInput.files[0];
  
    // Perform validation or additional processing if needed
  
    // Create a FileReader object to read the uploaded image file
    const reader = new FileReader();
  
    // Define a callback function to be executed when the FileReader has finished reading the file
    reader.onload = function (event) {
      // Get the URL of the uploaded image
      const imageUrl = event.target.result;
  
      // Create an img element and set its src attribute to the uploaded image URL
      const uploadedImage = document.createElement('img');
      uploadedImage.src = imageUrl;
  
      // Append the uploaded image element to a container on the page
      const imageContainer = document.querySelector('#uploaded-image-container');
      imageContainer.innerHTML = ''; // Clear previous uploaded images
      imageContainer.appendChild(uploadedImage);
    };
  
    // Read the uploaded image file as a Data URL
    reader.readAsDataURL(file);
  }
  
  // Add an event listener to the image upload form
  const uploadForm = document.querySelector('#upload-section form');
  uploadForm.addEventListener('submit', handleImageUpload);
  