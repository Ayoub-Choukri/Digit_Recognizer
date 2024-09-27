document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById('drawingCanvas');
    const context = canvas.getContext('2d');
    const messageContainer = document.getElementById('messageContainer');

    canvas.width = 400;
    canvas.height = 400;
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.lineWidth = 5;
    context.lineCap = "round";
    context.strokeStyle = "black";

    let drawing = false;
    let erasing = false;
    let eraserWidth = 10;


    


    function startDrawing(event) {
        drawing = true;
        draw(event);
    }

    function endDrawing() {
        drawing = false;
        context.beginPath();
    }

    function draw(event) {
        if (!drawing) return;

        context.lineWidth = erasing ? eraserWidth : document.getElementById("lineWidth").value;
        context.strokeStyle = erasing ? "white" : "black";

        context.lineTo(event.offsetX, event.offsetY);
        context.stroke();
        context.beginPath();
        context.moveTo(event.offsetX, event.offsetY);
    }

    canvas.addEventListener("mousedown", (event) => {
        startDrawing(event);
    });

    canvas.addEventListener("mouseup", () => {
        endDrawing();
    });

    canvas.addEventListener("mousemove", (event) => {
        draw(event);
    });

    document.getElementById("resetButton").addEventListener("click", () => {
        context.fillStyle = "white";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.beginPath();
        context.fillStyle = "black";

        fetch('/reset-test-images', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            displayMessage(data.message, true);
        })
        .catch((error) => {
            console.error('Error:', error);
            displayMessage('Erreur lors de la réinitialisation.', false);
        });
    });

    document.getElementById('ButtonAccueil').addEventListener("click", () => {
        window.location.href = "/";
    }
    );
    document.getElementById("eraserButton").addEventListener("click", () => {
        erasing = true;
    });

    document.getElementById("drawButton").addEventListener("click", () => {
        erasing = false;
    });

    document.getElementById("lineWidth").addEventListener("input", (event) => {
        if (!erasing) {
            context.lineWidth = event.target.value;
        }
    });

    document.getElementById("eraserWidth").addEventListener("input", (event) => {
        if (erasing) {
            eraserWidth = event.target.value;
        }
    });

    document.getElementById("saveButton").addEventListener("click", () => {
        saveImage();
    });


    document.getElementById("predictButton").addEventListener("click", () => {
        PredictImage();
    }

    );

    function PredictImage() {
        // Save the image first

        saveImage();

        // Scroll to predictionContainer
        document.getElementById("predictionContainer").scrollIntoView({ behavior: 'smooth' });


        fetch('/predict-image', {
            method: 'GET',
        })

        .then(response => response.json())

        .then(data => {
            displayMessage2(data.prediction);
        }

        )

        .catch((error) => {
            console.error('Error:', error);
            displayMessage2('Erreur lors de la prédiction de l\'image.');
        }

        
        );

    }


    function saveImage() {
        const image = canvas.toDataURL("image/png");

        // Scroll to predictionContainer
        document.getElementById("messageContainer").scrollIntoView({ behavior: 'smooth' });
        fetch('/save-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: image }),
        })
        .then(response => response.json())
        .then(data => {
            displayMessage(data.message, true);
        })
        .catch((error) => {
            console.error('Error:', error);
            displayMessage('Erreur lors de l\'enregistrement de l\'image.', false);
        });
    }

    function displayMessage(message, success) {
        messageContainer.textContent = message;
        messageContainer.style.color = success ? 'green' : 'red';
    }

    function displayMessage2(message) {
        predictionContainer.textContent = message;
        predictionContainer.style.color = 'green' ;
    }
});




