import threading
from jass.strategies.interfaces.trump_strategy_game_observation import TrumpStrategyGameObservation
from jass.game.game_observation import GameObservation
from jass.game import const
from jass.utils.rule_based_agent_util import calculate_trump_selection_score
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
# Beispiel: lade & predict (vollständiges Keras model in .h5)
from pathlib import Path
import numpy as np
from tensorflow import keras

class NeuronalNet(TrumpStrategyGameObservation):
    """
    Trump selection strategy that chooses trump based on a trained neural network model.
    Model and Data was trained on GPU Cluster from HSLU with .csv file
    This version works with GameObservation instead of GameState.

    This strategy doesn't return PUSH.
    """
    
    _model = None
    _lock = threading.Lock()
    _model_path = Path("/home/fabian/Documents/InformatikVault/Semester5/DL4G/AgentCode/jass-kit-py/neuronal-model")

    @classmethod
    def _ensure_model(cls):
        with cls._lock:
            if cls._model is None:
                if not cls._model_path.exists():
                    raise FileNotFoundError(cls._model_path)
                cls._model = keras.models.load_model(cls._model_path)

    @classmethod
    def clear_model(cls):
        with cls._lock:
            if cls._model is not None:
                try:
                    del cls._model
                except Exception:
                    pass
                cls._model = None
                keras.backend.K.clear_session()
                import gc; gc.collect()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        cardsOfCurrentPlayer = obs.hand
        print("Cards of current player (one-hot):", cardsOfCurrentPlayer)

        current_player_index = obs.player_view
        dealer_index = obs.dealer
        
        forehand_element_input_element = 0
        #check if current player is forehand and can still push
        if current_player_index == (dealer_index + 1) % 4 :
            current_player_is_forehand = True
            forehand_element_input_element = 1
        else:
            current_player_is_forehand = False
            forehand_element_input_element = 0

        cards_with_batch_dim = np.expand_dims(cardsOfCurrentPlayer, axis=0) # Form: (1, 36)

        # 2. Das zusätzliche Feature (forehand_element) von () auf (1, 1) umformen
        forehand_feature = np.array([[forehand_element_input_element]]) # Form: (1, 1)

        # 3. Beide Arrays horizontal (axis=1) zu einem einzigen Input-Array verbinden
        # Das Ergebnis ist ein Array der Form (1, 37)
        input_data = np.concatenate([cards_with_batch_dim, forehand_feature], axis=1)
        print("Input array into Neural net element:", input_data)
        
        if obs.trump == -1 and current_player_is_forehand:
            # Forehand must choose trump, use model
            self._ensure_model()

            prediction = self._model.predict(input_data, verbose=0)
            # neuronal_model = keras.models.load_model('/home/fabian/Documents/InformatikVault/Semester5/DL4G/AgentCode/jass-kit-py/neuronal-model/model.weights.h5')
            # prediction = neuronal_model.predict(cardsOfCurrentPlayer)

            # Predict trump scores
            print("Predictions:", prediction)
            if np.argmax(prediction, axis=1) == 6:
                # return random
                return const.PUSH
            else:
                return np.argmax(prediction, axis=1)

        elif obs.trump == -1 and not current_player_is_forehand:
            self._ensure_model()
            prediction = self._model.predict(input_data, verbose=0)

            if np.argmax(prediction, axis=1) == 6:
                # return random
                return const.DIAMONDS
            else:
                return np.argmax(prediction, axis=1)

        else:
            # Trump already chosen, return it
            prediction = obs.trump
        return prediction
        