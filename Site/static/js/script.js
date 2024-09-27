    let WEBSITE_API_URL = "http://127.0.0.1:5000"
    let API_MODEL_URL = "http://127.0.0.1:5001"

document.addEventListener('DOMContentLoaded', function() {
    var startDrawingButton = document.getElementById('startDrawingButton');
    startDrawingButton.addEventListener('click', async function() {
        var model = document.getElementById("model").value;
        if (model) {
            console.log("Specifying model: " + model);

            let response = await  fetch('http://127.0.0.1:5000/Specify_Model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_name: model })
            })

            if (response.ok) {
                console.log("Model specified successfully.");
            } else {
                const errorText = await response.text();
                console.log("Failed to specify model. Error: " + errorText);
            }

            window.location.href = "drawing";
        } else {
            alert("Veuillez sélectionner un modèle avant de continuer.");
        }
    });
});
