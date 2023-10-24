let wins = 0;
let losses = 0;
let draws = 0;

function makeChoice(playerChoice) {
    // Clear any previous highlights
    clearHighlights();

    // Send the player's choice to the server
    fetch(`/play/${playerChoice}`)
        .then(response => response.json())
        .then(data => {
            let aiChoice = data.ai_choice;

            // Determine the result and highlight the player's choice accordingly
            if (playerChoice === aiChoice) {
                highlightChoice(playerChoice, 'draw', aiChoice);
            } else if (
                (playerChoice === 'rock' && aiChoice === 'scissors') ||
                (playerChoice === 'scissors' && aiChoice === 'paper') ||
                (playerChoice === 'paper' && aiChoice === 'rock')
            ) {
                highlightChoice(playerChoice, 'win', aiChoice);
            } else {
                highlightChoice(playerChoice, 'lose', aiChoice);
            }

            // Update the tally
            updateTally(playerChoice, aiChoice);
        });
}

function clearHighlights() {
    let choices = ['rock', 'paper', 'scissors'];
    choices.forEach(choice => {
        let element = document.getElementById(choice);
        element.classList.remove('win', 'lose', 'draw');
    });
}

function highlightChoice(playerChoice, result, aiChoice) {
    let playerElement = document.getElementById(playerChoice);
    let aiElement = document.getElementById(aiChoice);

    if (result === 'win') {
        playerElement.classList.add('win');
        aiElement.classList.add('lose');
    } else if (result === 'lose') {
        playerElement.classList.add('lose');
        aiElement.classList.add('win');
    } else {
        playerElement.classList.add('draw');
        aiElement.classList.add('draw');
    }
}


function updateTally(playerChoice, aiChoice) {
    if (playerChoice === aiChoice) {
        let drawsEl = document.getElementById('draws');
        draws += 1;
        drawsEl.textContent = draws;
    } else if (
        (playerChoice === 'rock' && aiChoice === 'scissors') ||
        (playerChoice === 'scissors' && aiChoice === 'paper') ||
        (playerChoice === 'paper' && aiChoice === 'rock')
    ) {
        let winsEl = document.getElementById('wins');
        wins += 1;
        winsEl.textContent = wins;
    } else {
        let lossesEl = document.getElementById('losses');
        losses += 1;
        lossesEl.textContent = losses;
    }
}

