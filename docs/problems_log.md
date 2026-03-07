# Problems Log
*Short summaries of issues and fixes will be logged here.*

## Issue: ModuleNotFoundError: No module named 'services'
*   **Cause:** The terminal was opened in the `Mushrooms` root directory (which contains the virtual environment and the `Mushroom` project folder), but the command was executed expecting to be *inside* the `Mushroom` project folder where `services` is located. Since Python couldn't find `services` in the parent `Mushrooms` directory, it threw an error.
*   **Proposed Solution:** Change the working directory into the project folder (`cd Mushroom`) before running the uvicorn command, so Python's path aligns with the folder structure.
