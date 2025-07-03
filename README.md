# FoodVision - Hugging Face Spaces

This is a Food Image Classifier deployed with FastAPI on Hugging Face Spaces.

## Project Structure
```
/your-project-folder
├── app/
│   └── main.py         # FastAPI backend
├── static/
│   └── index.html      # HTML frontend
├── model/              # TensorFlow SavedModel files
├── requirements.txt    # Python dependencies
└── README.md           # Project info
```

## Usage
- Upload a food image using the web interface.
- The backend predicts the food class using a TensorFlow model.
- The frontend fetches a recipe for the predicted class.

## Deployment
- This project is ready for Hugging Face Spaces (FastAPI backend + static frontend).
- All dependencies are listed in `requirements.txt`.

## Credits
- Built with FastAPI, TensorFlow, and Hugging Face Spaces. 