# Job Board Vision Assistant - Computer Vision & ML Project

## Description

Traditional job searching involves manually browsing through hundreds of job postings, reading each description, and trying to extract key information like salary, requirements, and company details. This process is time-consuming, repetitive, and prone to human error. Job seekers often miss important details or spend hours on applications that don't match their criteria.

<img src="https://i.imgur.com/Lo2QYAe.png" width="600">

Finding the right job is hard. You have to read through hundreds of job postings, figure out the salary, and see if you match the requirements. This takes forever and you might miss important details.

This project solves that problem. It uses computer vision to read job postings from screenshots and machine learning to understand what the job is about.

The system uses EasyOCR to read text from images, OpenCV to clean up the images, and Machine Learning to classify jobs and predict salaries.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Installation

**Option 1: Clone and Install (Local)**

### System Requirements
- **Python 3.8+** (Required)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/chandra122/job-board-vision-assistant.git
cd job-board-vision-assistant

# Install all the required packages
pip install -r requirements.txt

# Run the web interface
python app.py
```

### Platform Notes
- **Windows**:  Fully tested and working
- **macOS**:  Should work (Python libraries are cross-platform)
- **Linux**:  Should work (Python libraries are cross-platform)

**Note**: Some dependencies (like `pyreadline3`) are Windows-specific but won't affect functionality on Mac/Linux. The core functionality uses cross-platform libraries.

**Performance Optimizations**: The project includes Microsoft optimization libraries (`onnxruntime`, `openvino`) for 2-5x faster ML inference. These work on all platforms but provide the best performance on Windows with Intel/AMD processors.

**Option 2: Google Colab (No Installation) - RECOMMENDED**

### Step-by-Step Guide for Google Colab:

**Step 1: Download the Notebooks**
1. Go to [Google Drive](https://drive.google.com/drive/folders/1KHjGZ007rIcjnER6WAdWHPaV4ZURNxLr?usp=drive_link)
2. Download both files:
   - **ML_Testing_Notebook.ipynb** (160 KB)
   - **CV_Testing_Notebook.ipynb** (44 KB)

**Step 2: Open in Google Colab**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "Upload" and select the downloaded `.ipynb` files
3. Choose which notebook to run first (ML or CV)

**Step 3: Get Test Images (For CV Testing)**
For the Computer Vision notebook, you need LinkedIn job screenshots:

**Easy Option - Use Our Sample Images:**
1. Download from [ComputerVisionImages folder](https://drive.google.com/drive/folders/1KHjGZ007rIcjnER6WAdWHPaV4ZURNxLr?usp=drive_link)
2. Upload to `/content/` folder in Colab

**Your Own Images:**
1. Take screenshots of LinkedIn job postings
2. Save as `.png` files
3. Upload to Colab's `/content/` folder
4. Name them like: `job_card_1.png`, `linkedin_job.png`, etc.

**Step 4: Run the Notebooks**
1. **ML Testing**: Run all cells to see machine learning in action
2. **CV Testing**: Run all cells to see computer vision processing

### What Each Notebook Does:

**ML_Testing_Notebook.ipynb:**
- Tests job classification (Data Scientist, Software Engineer, etc.)
- Tests salary prediction models
- Shows performance metrics and visualizations
- No images needed - uses sample data

**CV_Testing_Notebook.ipynb:**
- Tests OCR text extraction from job screenshots
- Processes real LinkedIn job postings
- Shows before/after image processing
- **Requires job screenshot images**

## Project Structure

### Folder Organization
```
job-board-vision-assistant/
├── src/                                    # Core source code
│   ├── automation/
│   │   └── fully_automated_pipeline.py    # Main automation workflow
│   ├── computer_vision/
│   │   └── enhanced_job_analyzer.py       # OCR and image processing
│   ├── data_collection/
│   │   ├── linkedin_scraper.py            # LinkedIn job scraping
│   │   ├── kaggle_collector.py            # Kaggle dataset collection
│   │   └── screenshot_capture.py          # Screenshot capture functionality
│   ├── ml_models/
│   │   └── job_classifier.py              # Machine learning models
│   └── utils/
│       ├── email_sender.py                # Email automation
│       ├── linkedin_folder_manager.py     # File organization
│       └── microsoft_optimizations.py     # Performance optimizations
├── templates/                              # Web interface templates
│   ├── index.html                         # Main dashboard
│   ├── workflow.html                      # Workflow selection
│   ├── results.html                       # Results display
│   └── visualizations.html                # Data visualizations
├── config/                                # Configuration files
│   ├── cron_config.json                   # Scheduling settings
│   └── email_config.json                  # Email settings
├── data/                                  # Data storage
│   ├── raw/                               # Raw screenshots and data
│   ├── processed/                         # Processed job data
│   └── outputs/                           # Analysis results
├── app.py                                 # Flask web application
├── start_web_ui.py                        # Web UI startup script
├── run_scraper.py                         # Standalone scraper script
├── requirements.txt                       # Python dependencies
└── sample_data.json                       # Sample data for testing
```

### Key Files Explained

**Core Application Files:**
- **`app.py`** - Main Flask web application (start here for web interface)
- **`start_web_ui.py`** - Web UI startup script
- **`run_scraper.py`** - Standalone scraper script for data collection
- **`requirements.txt`** - All Python packages needed

**Source Code (`src/` folder):**
- **`fully_automated_pipeline.py`** - Complete automation workflow
- **`enhanced_job_analyzer.py`** - Computer vision and OCR processing
- **`job_classifier.py`** - Machine learning models for job analysis
- **`linkedin_scraper.py`** - LinkedIn job posting collection
- **`kaggle_collector.py`** - Dataset collection from Kaggle

**Web Interface (`templates/` folder):**
- **`index.html`** - Main dashboard with project overview
- **`workflow.html`** - Workflow selection and configuration
- **`results.html`** - Display analysis results and visualizations

**Configuration (`config/` folder):**
- **`email_config.json`** - Email settings for notifications
- **`cron_config.json`** - Scheduling configuration

**Data Storage (`data/` folder):**
- **`raw/`** - Original LinkedIn screenshots and scraped data
- **`processed/`** - Cleaned and structured job data
- **`outputs/`** - Final analysis results and reports

### Requirements

- **Python 3.8+** (Required)
- **All dependencies** are listed in `requirements.txt`

### What Gets Installed

The `requirements.txt` includes all necessary libraries:
- **easyocr** - OCR text extraction from images
- **opencv-python** - Image processing and enhancement
- **scikit-learn** - Machine learning models and predictions
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **flask** - Web interface and API
- **matplotlib** - Data visualization
- **And more...** (see requirements.txt for complete list)

## Usage

### The Story: From Screenshot to Smart Analysis

Imagine you're scrolling through LinkedIn and you see a job posting that looks interesting. Instead of spending 10 minutes reading through all the details, you just take a screenshot and let our system do the work.

**You Take a Screenshot**
You see a job posting like this and think "This looks good, but I need to know more about it":

<img src="https://i.imgur.com/Q84ry07.png" width="400" height="200">

**From Screenshot to Structured Data**
Our computer vision system takes your screenshot and extracts all the important information, turning it into organized data:

<img src="https://i.imgur.com/FEqQSsw.png" width="800" height="200">

**From Text to Smart Analysis**
The extracted text then goes through our machine learning pipeline to understand what the job is about and predict key details and send the email output :

<img src="https://i.imgur.com/I924k6R.png" width="800" height="200">

**The Magic Happens in Seconds**
What used to take you 10 minutes of reading now takes our system 30 seconds to analyze and give you the key information you need to decide if you want to apply.

### Key Features

**Computer Vision Pipeline:**
- Automatic text extraction from job posting screenshots
- Image preprocessing and enhancement
- OCR confidence scoring and validation
- Structured data extraction and cleaning

**Machine Learning Analysis:**
- Job category classification (Data Scientist, Software Engineer, etc.)
- Seniority level prediction (Entry, Mid-level, Senior)
- Salary range estimation
- Skills and requirements analysis

**Web Interface:**
- User-friendly Flask web application
- Real-time processing status updates
- Interactive results visualization
- Multiple workflow options

**Performance Optimizations:**
- Microsoft ONNX Runtime for 2-5x faster ML inference
- Intel OpenVINO for CPU optimization
- Microsoft Windows ML acceleration (Windows 10+)
- Cross-platform performance improvements


### Technical Architecture

**Data Collection:**
- LinkedIn job scraping (automated)
- Kaggle dataset integration
- Real-time job posting processing

**Computer Vision:**
- EasyOCR for text extraction
- OpenCV for image preprocessing
- Custom text cleaning algorithms
- Confidence scoring and validation

**Machine Learning:**
- Random Forest classifiers for job categorization
- TF-IDF vectorization for text analysis
- Salary prediction using regression models
- Performance metrics and validation

**Web Interface:**
- Flask-based REST API
- Real-time status updates
- Interactive visualizations
- Multiple workflow support

## References

### Tutorials and Learning Resources
- [Nicholas Renotte - Computer Vision Tutorials](https://www.youtube.com/c/NicholasRenotte)
- [Nicholas Renotte - Machine Learning with Python](https://www.youtube.com/c/NicholasRenotte)

### Libraries and Frameworks
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR) - OCR text extraction
- [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/) - Computer vision processing
- [scikit-learn Documentation](https://scikit-learn.org/stable/) - Machine learning algorithms
- [Flask Documentation](https://flask.palletsprojects.com/) - Web application framework

### Microsoft Optimization Libraries
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference acceleration
- [Intel OpenVINO](https://docs.openvino.ai/) - CPU optimization for deep learning
- [Microsoft Windows ML](https://docs.microsoft.com/en-us/windows/ai/) - Windows 10+ ML acceleration

### Data Sources
- [LinkedIn Job Postings](https://www.linkedin.com/jobs/) - Job posting data source
- [Kaggle Datasets](https://www.kaggle.com/datasets) - Job market datasets for training

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

### What this means:
-  **Free to use** - You can use this project for any purpose
-  **Free to modify** - You can change the code as needed
-  **Free to distribute** - You can share the project with others
-  **Share alike** - If you distribute modified versions, you must use the same license
-  **Source code** - You must make the source code available when distributing

### For commercial use:
This project can be used commercially, but any modifications or distributions must also be licensed under GPL v3.0.