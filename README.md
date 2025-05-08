# Disease Prediction Model with LIME Explanations

## Overview
This project develops a machine learning model to predict the likelihood of three diseases: **diabetes**, **lung cancer**, and **non-alcoholic fatty liver disease (NAFLD)** using a `RandomForestClassifier`. To enhance interpretability, we employ **LIME (Local Interpretable Model-agnostic Explanations)** to provide insights into model predictions. 

Additionally, a **frontend interface** enables patients to upload images of their prescriptions. These are converted into text using **Google Gemini's image-to-text API**, and the extracted text is then processed by the model to generate predictions.

---

## Features
- **Disease Prediction**: Uses `RandomForestClassifier` to predict the likelihood of diabetes, lung cancer, and NAFLD.
- **LIME Explanations**: Integrates LIME for model-agnostic interpretation of predictions.
- **Frontend Interface**: Allows patients to upload prescription images.
- **Text-based Prediction**: Converts prescription images to text using Google Gemini and feeds it into the ML model.

---

## Technical Requirements
- Python 3.x
- `scikit-learn`
- `lime`
- Google Gemini API
- `flask`

---

## Model Details
- **RandomForestClassifier**: Trained on features extracted from prescription text to predict disease likelihood.
- **LIME**: Provides local explanations for each individual prediction, improving transparency and trust.

---

## Usage
1. Upload a prescription image through the frontend.
2. Image is processed using **Google Gemini**, converting it to text.
3. Extracted text is input into the ML model.
4. Model outputs predictions for diabetes, lung cancer, and NAFLD.
5. **LIME explanations** are displayed alongside predictions.

---

## Future Work
- Improve accuracy and robustness of the ML model.
- Integrate with Electronic Health Record (EHR) systems.
- Extend support to other diseases and conditions.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- **Marco Ribeiro** – for developing LIME.
- **Google** – for the Gemini image-to-text functionality.
