## Running test_ebda.ipynb Locally

Follow these steps after cloning the repository:

1. **Install Python (recommended: 3.11+)**
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   pip install jupyter
   ```
4. **(Optional) Install additional dependencies for notebooks:**
   If you use conda, you may want to run:
   ```bash
   conda install pandas numpy
   ```
5. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
6. **Open `notebook/test_ebda.ipynb` in the Jupyter interface.**
7. **Run the cells in order.**

**Notes:**
- Make sure you have the required datasets in the expected locations (see notebook code for paths).
- If you encounter missing packages, install them using `pip install <package>`.
- If you use Docker, you can run Jupyter in a container (see Dockerfile for details).
