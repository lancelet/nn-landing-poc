import Reveal from 'reveal.js';
import 'reveal.js/dist/reveal.css'
import 'reveal.js/dist/theme/black.css'

// Initialize the Reveal.js presentation framework.
Reveal.initialize();

// Add listener to go fullscreen when the fullscreen buttons are pressed.
const fullscreenButtons = document.getElementsByClassName("fullscreenButton");
const fullscreenButtonsArray = Array.from(fullscreenButtons);
fullscreenButtonsArray.forEach(function (element) {
    element.addEventListener("click", function () {
        const revealElement = document.querySelector(".reveal");

        if (revealElement.requestFullscreen) {
            revealElement.requestFullscreen();
        } else if (revealElement.mozRequestFullScreen) {
            // Firefox
            revealElement.mozRequestFullScreen();
        } else if (revealElement.webkitRequestFullscreen) {
            // Chrome, Safari, Opera
            revealElement.webkitRequestFullscreen();
        } else if (revealElement.msRequestFullscreen) {
            // IE/Edge
            revealElement.msRequestFullscreen();
        }
    });
});