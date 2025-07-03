document.addEventListener("DOMContentLoaded", function() {
  const goToPredictionsButton = document.getElementById("go-to-predictions");
  if (goToPredictionsButton) {
    goToPredictionsButton.addEventListener("click", function() {
      window.location.href = "/predict";
    });
  }
});
