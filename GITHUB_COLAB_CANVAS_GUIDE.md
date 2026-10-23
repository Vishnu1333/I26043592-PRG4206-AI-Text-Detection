# GitHub, Google Colab, and Canvas Submission Guide

## A. Publish the project on GitHub

1. Sign in to GitHub and create a new repository named `I26043592-PRG4206-AI-Text-Detection`.
2. For a public coursework link, set visibility to **Public** only if you are comfortable sharing the coursework publicly and your university rules permit it. Otherwise use Private and confirm the lecturer can access it.
3. Upload the **contents** of the `I26043592_Vishnu_Vardhan_Code` folder to the repository (not the ZIP as the only repository file). Include the source folders, models, tests, documentation, the Colab notebook, `requirements-colab.txt`, and the outputs folder. Do not upload a `.venv`, credentials, tokens, `.DS_Store`, or cache directories.
4. Wait for GitHub to finish uploading. Open `README.md` in the repository and make sure it displays, and click the notebook at `notebooks/00_PRG4206_Colab_Run_and_Outputs.ipynb` to confirm it was uploaded.
5. Copy the real repository URL from the browser address bar. Do not use a placeholder link.

## B. Run the notebook in Google Colab

1. Open Google Colab and choose **File → Open notebook → GitHub**. Paste the actual repository URL or search for your username/repository, then open `notebooks/00_PRG4206_Colab_Run_and_Outputs.ipynb`. If GitHub lookup does not find it, use Colab's **Upload** tab and upload the `.ipynb` file directly.
2. Run the notebook cells in order. At the upload prompt, upload the code ZIP from Downloads. Use the clean project ZIP, not the original 1.5 GB folder/ZIP.
3. Let the cells finish. Review the displayed test summary, statistical verification, three-model predictions, and TXT/PDF/DOCX document parser checks. Fix any failures before claiming success.
4. Download `PRG4206_Project_Outputs_Summary.pdf` and `I26043592_Vishnu_Vardhan_Code.zip` from the last cell. The PDF's Colab status must reflect the actual run.
5. If you want the output PDF and updated notebook in GitHub as well, upload the generated PDF, `outputs/colab_runtime.json`, and any notebook with saved outputs back to the repository. Do not replace the code archive until you have checked the new ZIP.

## C. Canvas submission

- **Code assignment:** upload `I26043592_Vishnu_Vardhan_Code.zip` (or the lecturer-specified ZIP filename). If Canvas only accepts one file, use the ZIP; add the repository URL in submission comments if that control is offered, or add it to the README and/or a separate link field as your course requires.
- **Turnitin assignment:** use the report PDF specified by the lecturer, not the code ZIP. Keep the original Turnitin receipt separately as proof of the earlier submission.
- **Outputs PDF:** include `outputs/PRG4206_Project_Outputs_Summary.pdf` inside the code ZIP as requested. If your assignment page accepts only one ZIP, do not substitute the outputs PDF for the ZIP.
- **Presentation/video:** submit separately if requested by the coursework brief.

## D. What the output PDF does and does not show

The notebook verifies the three saved models and the stored held-out predictions; it is not a fresh re-training experiment. Its report is generated from current command results and a timestamped Colab runtime record. The report should show `PASS` only when the required checks pass; a `REVIEW REQUIRED` status means the errors must be reviewed. Document-level scores are aggregate chunk predictions, not independent document-level benchmark accuracy.
