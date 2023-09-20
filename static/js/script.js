function submitForm() {
    document.querySelector(".pixel-spinner").style.display = 'flex';
    document.getElementById("predict-button").click();
}

function submittext(){
    var pixelSpinner = document.querySelector(".pixel-spinner");
    pixelSpinner.style.display = 'flex';
}

// Add this JavaScript code after the HTML element with the data-ai-probability attribute
const progressBar = document.querySelector(".progress-bar");

// Get the AI probability value from the data attribute
const aiProbability = parseFloat(progressBar.getAttribute("data-ai-probability"));

// Calculate the gradient stops based on the AI probability
const average = Math.round(aiProbability * 100);
var color;
if(average > 80){
    color = 'rgb(179, 7, 7)'
}else if(average > 60){
    color = '#d47c34'
}else if(average > 40){
    color = '#f5cf43'
}else{
    color = '#5dc2ab'
}


// Update the content of the ::before pseudo-element with the AI probability
if(average < 40){
    progressBar.style.background = `
    radial-gradient(closest-side, white 79%, transparent 80% 100%),
    conic-gradient(${color} ${100-average}%, rgb(218, 217, 215) 0)`;
    progressBar.style.color = color
    progressBar.textContent = `${100-average}%`;
}else {
    progressBar.style.background = `
    radial-gradient(closest-side, white 79%, transparent 80% 100%),
    conic-gradient(${color} ${average}%, rgb(218, 217, 215) 0)`;
    progressBar.style.color = color
    progressBar.textContent = `${average}%`;
}

document.getElementById("result").style.display = 'flex';
if (window.innerWidth > 480) {
    // Apply the margin only for larger screens
    document.getElementById("input").style.marginLeft = '50px';
}


function openOverlay() {
    document.getElementById("overlay").style.display = "block";
}

// Function to close the overlay
function closeOverlay() {
    document.getElementById("overlay").style.display = "none";
}

function closeErrorOverlay() {
    document.getElementById("error-overlay").style.display = "none";
}

// Attach the openOverlay function to the "Report" label
document.getElementById("print").addEventListener("click", openOverlay);

function printReport() {
    document.querySelector(".buttons").style.display = "none"
    var printWindow = window.open('', '', 'width=800,height=600');
    printWindow.document.open();
    printWindow.document.write('<html><head><title>Printable Report</title>');
    printWindow.document.write('<link rel="stylesheet" type="text/css" href="/static/css/index.css" media="print">'); // Reference the CSS file directly
    printWindow.document.write('</head><body>');
    printWindow.document.write(document.getElementById('report-content').innerHTML);
    printWindow.document.write('</body></html>');
    document.querySelector(".buttons").style.display = "flex"
    printWindow.document.close();
    printWindow.print();
    printWindow.close();
}

