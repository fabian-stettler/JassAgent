# jass-kit-py

Base components to program agents for the game of jass (https://en.wikipedia.org/wiki/Jass)


# start locally deployed flask Server
(base) fabian@fabian-Yoga-7-16IAP7:~/Documents/InformatikVault/Semester5/DL4G/AgentCode/jass-kit-py$ PYTHONPATH=. python -m examples.service.player_service


# start a test game with two local random players and two players from the flask server
cd /home/fabian/Documents/InformatikVault/Semester5/RL4GAMES/AgentCode/jass-kit-py && python test_server_game.py


# publish local flask server with ngrok
- precondition: ngrok is set up and running // --> https://dashboard.ngrok.com/get-started/setup/linux
- ngrok http <exposed_port>  //e.g. 5000
- players should now be accessible through endpoints 
