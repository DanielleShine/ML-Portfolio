const navButton = document.getElementById("nav-button");
const leftNav = document.getElementById("left-nav");

navButton.addEventListener("click", function() {
  leftNav.classList.toggle("show");
});