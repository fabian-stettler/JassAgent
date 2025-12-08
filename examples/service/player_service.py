# HSLU
#
# Created by Thomas Koller on 7/30/2020
#
"""
Example how to use flask to create a service for one or more players
"""
import logging
import time
import uuid
from logging.handlers import RotatingFileHandler

from flask import request, g

from examples.arena.arena_play import MyAgent
from jass.service.player_service_app import PlayerServiceApp
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.rule_based_agent import RuleBasedAgent
from jass.agents.agent_mcts_observation import AgentByMCTSObservation
from jass.agents.agent_mcts_observation_gpu import AgentByMCTSObservationGPU


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    # Configure logging once (avoid duplicate handlers when the reloader imports the module twice)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Basic logging to console
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

        # Rotating file handler so we can inspect request/response logs
        fh = RotatingFileHandler('player_service.log', maxBytes=10 * 1024 * 1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        root_logger.addHandler(fh)

        # Reduce verbosity of noisy third-party loggers (debugpy, watchdog/inotify, pydevd)
        noisy_loggers = ('watchdog', 'inotify_simple', 'debugpy', 'pydevd', 'pydevd_*')
        for nl in noisy_loggers:
            try:
                logging.getLogger(nl).setLevel(logging.WARNING)
            except Exception:
                # If the logger isn't present yet, ignore
                pass

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('AgentRandomSchieber', AgentRandomSchieber())
    app.add_player('AgentRandomSchieber2', AgentRandomSchieber())
    app.add_player('RuleBasedAgent', RuleBasedAgent())
    app.add_player('RuleBasedAgent2', RuleBasedAgent())
    app.add_player('MCTSObservationAgent', AgentByMCTSObservation(samples=10, simulations_per_sample=300, time_limit_sec=7))
    app.add_player('MCTSObservationAgent2', AgentByMCTSObservation(samples=10, simulations_per_sample=300, time_limit_sec=7))  
    app.add_player('MCTSObservationAgentGPU', AgentByMCTSObservationGPU(samples=8, simulations_per_sample=200, time_limit_sec=7))
    app.add_player('MCTSObservationAgentGPU2', AgentByMCTSObservationGPU(samples=8, simulations_per_sample=200, time_limit_sec=7))  

    # Request / response logging for debugging third-party integrations (captures headers and body)
    @app.before_request
    def _log_request():
        # lightweight request id to correlate logs
        g.request_start_time = time.time()
        g.request_id = str(uuid.uuid4())[:8]
        try:
            body = request.get_data(as_text=True)
        except Exception:
            body = '<unreadable>'
        # Don't log huge bodies fully; truncate to 8k
        if body and len(body) > 8192:
            body = body[:8192] + '...[truncated]'
        logging.getLogger(__name__).debug(f"[{g.request_id}] IN  {request.method} {request.path} headers={dict(request.headers)} body={body}")

    @app.after_request
    def _log_response(response):
        duration_ms = (time.time() - g.get('request_start_time', time.time())) * 1000.0
        logging.getLogger(__name__).debug(f"[{g.get('request_id','-')} ] OUT {request.method} {request.path} status={response.status} duration_ms={duration_ms:.1f}")
        return response

    return app


if __name__ == '__main__':
   app = create_app()
   # Make server accessible from other machines in the network
   app.run(host='0.0.0.0', port=5000, debug=True)

