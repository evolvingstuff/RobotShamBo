from bottle import Bottle, request, response, static_file, run
from beaker.middleware import SessionMiddleware
import uuid
from src.utils import load_champion
from config.config import choice_to_id, id_to_choice

app = Bottle()

# Set up the session options
session_opts = {
    'session.type': 'memory',
    'session.cookie_expires': 3600,  # 1 hour
    'session.auto': True
}

@app.route('/')
def index():
    return static_file('index.html', root='static/html/')

@app.route('/js/<filename>')
def serve_js(filename):
    return static_file(filename, root='static/js/')

@app.route('/css/<filename>')
def serve_css(filename):
    return static_file(filename, root='static/css/')

@app.route('/images/<filename>')
def serve_images(filename):
    return static_file(filename, root='static/images/')

@app.route('/play/<player_choice>', method='GET')
def play_rps(player_choice):
    s = request.environ['beaker.session']
    user_id = s.get('user_id')

    # If the user doesn't have an ID, create one and initialize their AI instance
    if not user_id:
        user_id = str(uuid.uuid4())
        s['user_id'] = user_id
        s['ai'] = load_champion()
        s['previous_player_choice'] = None
        s.save()

    # Retrieve the AI instance from the session
    user_ai = s['ai']
    previous_player_choice = s['previous_player_choice']

    if player_choice not in ["rock", "paper", "scissors"]:
        response.status = 400
        return {"error": "Invalid choice"}
    if previous_player_choice is None:
        ai_choice_id = user_ai.move(None)
    else:
        ai_choice_id = user_ai.move(choice_to_id[previous_player_choice])
    s['previous_player_choice'] = player_choice
    return {
        "ai_choice": id_to_choice[ai_choice_id]
    }


if __name__ == "__main__":
    app_with_sessions = SessionMiddleware(app, session_opts)
    run(app=app_with_sessions, host='0.0.0.0', port=8080)
