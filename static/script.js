class ImageRecognitionApp {
    constructor() {
        this.selectedFile = null;
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.fileInfo = document.getElementById('fileInfo');
        this.predictBtn = document.getElementById('predictBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.errorMessage = document.getElementById('errorMessage');
        this.previewImage = document.getElementById('previewImage');
    }

    setupEventListeners() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.imageInput.click();
        });

        // File input change
        this.imageInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            this.handleFileSelect(e.dataTransfer.files[0]);
        });

        // Predict button
        this.predictBtn.addEventListener('click', () => {
            this.predictImage();
        });

        // Remove file button
        document.getElementById('removeFile').addEventListener('click', () => {
            this.clearSelection();
        });

        // New prediction button
        document.getElementById('newPredictionBtn').addEventListener('click', () => {
            this.resetApp();
        });

        // Retry button
        document.getElementById('retryBtn').addEventListener('click', () => {
            this.hideError();
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB.');
            return;
        }

        this.selectedFile = file;
        this.showFileInfo(file);
        this.predictBtn.disabled = false;
    }

    showFileInfo(file) {
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        
        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
        
        this.fileInfo.style.display = 'flex';
        this.uploadArea.style.display = 'none';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async predictImage() {
        if (!this.selectedFile) return;

        this.showLoading();
        
        const formData = new FormData();
        formData.append('image', this.selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Prediction failed');
            }
        } catch (error) {
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        // Show preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
        };
        reader.readAsDataURL(this.selectedFile);

        // Display main prediction
        document.getElementById('mainPrediction').textContent = result.prediction;
        document.getElementById('mainConfidence').textContent = `${result.confidence.toFixed(1)}%`;

        // Display top 3 predictions
        const predictionsList = document.getElementById('predictionsList');
        predictionsList.innerHTML = '';
        
        result.all_predictions.forEach((pred, index) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <span class="pred-class">${pred.class}</span>
                <span class="pred-confidence">${pred.confidence.toFixed(1)}%</span>
            `;
            predictionsList.appendChild(item);
        });

        this.resultsSection.style.display = 'block';
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    showError(message) {
        document.getElementById('errorText').textContent = message;
        this.errorMessage.style.display = 'flex';
    }

    hideError() {
        this.errorMessage.style.display = 'none';
    }

    clearSelection() {
        this.selectedFile = null;
        this.imageInput.value = '';
        this.fileInfo.style.display = 'none';
        this.uploadArea.style.display = 'block';
        this.predictBtn.disabled = true;
    }

    resetApp() {
        this.clearSelection();
        this.resultsSection.style.display = 'none';
        this.hideError();
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageRecognitionApp();
});
