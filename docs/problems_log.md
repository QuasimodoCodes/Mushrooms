# Problems Log
*Short summaries of issues and fixes will be logged here.*

## Issue: TypeError: predict_image() takes 1 positional argument but 2 were given
*   **Cause:** We refactored `predict.py` to route images to the new Vision API, removing the need for a local `model_path` argument. However, we forgot to update the main `app.py` UI file to stop passing that obsolete second argument.
*   **Proposed Solution:** Update `app.py` to call `predict_image(image)` correctly, and remove the now-redundant local model weights existence check from the UI.

## Issue: ModuleNotFoundError: No module named 'services'
*   **Cause:** The terminal was opened in the `Mushrooms` root directory (which contains the virtual environment and the `Mushroom` project folder), but the command was executed expecting to be *inside* the `Mushroom` project folder where `services` is located. Since Python couldn't find `services` in the parent `Mushrooms` directory, it threw an error.
*   **Proposed Solution:** Change the working directory into the project folder (`cd Mushroom`) before running the uvicorn command, so Python's path aligns with the folder structure.
