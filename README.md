# AI Code Analyzer

AI Code Analyzer is a machine learning-powered web application that reviews source code, detects bugs, estimates complexity, and evaluates overall code quality in real time.

The project combines rule-based static analysis with Machine Learning techniques to help developers identify mistakes, improve code quality, and better understand performance implications.

 Live Demo: https://ai-code-analyzer-i6je.onrender.com/

---

## What the Project Does

Users can paste code directly into the web interface and instantly receive detailed analysis from multiple intelligent modules.

### Bug Detection Engine
The analyzer scans code line-by-line using 50+ custom detection rules designed to identify:
- Security vulnerabilities
- Logical bugs
- Performance issues
- Poor coding practices
- Common beginner mistakes
- Code quality problems

Each detected issue includes:
- Severity level (High / Medium / Low)
- Description of the problem
- Suggested fix or improvement

---

## Complexity Analysis

The system estimates:
- Time Complexity
- Space Complexity
- Overall Code Quality Score

It can identify common complexity patterns such as:
- O(1)
- O(log n)
- O(n)
- O(n²)
- O(n³)
- O(2^n)

The analysis is based on loop detection, nesting depth, recursion tracking, and pattern recognition.

---

## Machine Learning Classifier

The application includes a trained Multinomial Naive Bayes model that predicts whether submitted code is:
- Good Code
- Needs Improvement

The classifier also provides a confidence score for each prediction.

### Model Performance
The ML model was evaluated using 5-Fold Stratified Cross-Validation on 151 multilingual code samples.

| Metric | Score |
|---|---|
| Accuracy | 85.43% |
| Precision | 81.73% |
| Recall | 96.59% |
| F1 Score | 88.54% |

---

## Supported Languages

- Python
- JavaScript
- Java
- C++
- SQL

---

## Key Features

- Real-time code analysis
- AI-powered quality prediction
- Automatic bug detection
- Complexity estimation
- Auto-fix suggestions
- Language mismatch detection
- Interactive "Chat with Code" feature
- Fully offline processing (no external APIs)

---

## Technologies Used

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes

### Frontend
- HTML
- CSS
- JavaScript

### Deployment
- Render

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/khushi-solanki09/AI_code_Analyzer.git
cd AI_code_Analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

### 4. Open in browser

```bash
http://localhost:5000
```

---

## Future Improvements

- AST-based code analysis
- Deep Learning-powered recommendations
- Additional language support
- GitHub integration
- Real-time collaborative review

---

## Author

Khushi Singh Solanki