    let WEBSITE_API_URL = "http://127.0.0.1:5000"
    let API_MODEL_URL = "http://127.0.0.1:5001"

document.addEventListener('DOMContentLoaded', function() {
    var startDrawingButton = document.getElementById('startDrawingButton');
    startDrawingButton.addEventListener('click', function() {
        var model = document.getElementById("model").value;
        if (model) {

            fetch(API_MODEL_URL+'/Specify_Model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_name: model })
            })
                .then(response => response.json())
                .then(data => {
                    // Handle the response from the API
                    console.log(data);
                })
                .catch(error => {
                    // Handle any errors that occur during the request
                    console.error(error);
                });


            window.location.href = "drawing";
        } else {
            alert("Veuillez sélectionner un modèle avant de continuer.");
        }
    });
});
