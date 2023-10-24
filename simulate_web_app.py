from bottle import route, run, request, response, static_file
from utils import load_champion
from config import choice_to_id, id_to_choice

ai = load_champion()

# Route to serve the main page
@route('/')
def index():
    return static_file('index.html', root='static/html/')

# Route to serve JavaScript files
@route('/js/<filename>')
def serve_js(filename):
    return static_file(filename, root='static/js/')

# Route to serve CSS files
@route('/css/<filename>')
def serve_css(filename):
    return static_file(filename, root='static/css/')

# Route to serve images
@route('/images/<filename>')
def serve_images(filename):
    return static_file(filename, root='static/images/')

# Game logic
@route('/play/<player_choice>', method='GET')
def play_rps(player_choice):
    if player_choice not in ["rock", "paper", "scissors"]:
        response.status = 400
        return {"error": "Invalid choice"}
    ai_choice_id = ai.move(choice_to_id[player_choice])
    return {
        "ai_choice": id_to_choice[ai_choice_id]
    }


if __name__ == "__main__":
    """
    Play against current champion on localhost
    """
    run(host='localhost', port=8080)
