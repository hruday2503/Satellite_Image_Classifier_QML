document.getElementById('imageInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show image preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const imgPreview = document.getElementById('imagePreview');
        imgPreview.innerHTML = '<img src="' + e.target.result + '" alt="Uploaded Image" />';
    };
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append('image', file);

    // Clear previous result
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Classifying...';

    // Send image to backend API
    fetch('http://127.0.0.1:5000/classify', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.textContent = 'Error: ' + data.error;
        } else {
            resultDiv.textContent = 'Predicted Class: ' + data.predicted_class + ' (Confidence: ' + (data.confidence * 100).toFixed(2) + '%)';
        }
    })
    .catch(error => {
        resultDiv.textContent = 'Error: ' + error.message;
    });
});
