document.addEventListener('DOMContentLoaded', function() {
    var startDrawingButton = document.getElementById('startDrawingButton');
    startDrawingButton.addEventListener('click', function() {
        var model = document.getElementById("model").value;
        if (model) {
            window.location.href = "drawing";
        } else {
            alert("Veuillez sélectionner un modèle avant de continuer.");
        }
    });
});
