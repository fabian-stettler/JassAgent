import numpy as np

'''
Utilities for rule based agent
'''

# score if the color is trump
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0,]
# score if uneufe is selected (all colors)
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

def check_puur_with_four(hand, lowerBorderIndice, upperBorderIndice, puurIndice):
    kindCount = 0
    for i in range (lowerBorderIndice, upperBorderIndice):
        if hand[i] == 1: 
            kindCount+=1
    
    # WICHTIG: kindCount enthält ALLE Karten dieser Farbe (inklusive Jack)
    # Die Regel besagt: Jack + 3 oder mehr ANDERE Karten
    # Also: Wenn Jack vorhanden ist, brauchen wir mindestens 4 Karten total
    # (1 Jack + 3 andere = 4 total)
    
    if hand[puurIndice] == 1 and kindCount >= 4:
        return 1
    else:
        return 0

    

def havePuurWithFour(hand: np.ndarray) -> np.ndarray:
    result = np.zeros(4, dtype=int)
    # Korrekte Kartengrenzen: jede Farbe hat 9 Karten
    puurIndices = [3, 12, 21, 30]
    kind_border_indices = [0, 9, 18, 27, 36]
        
    #split cards of player into for subsets (for each kind one) and check for rule application

    for i in range(4):
        result[i] = check_puur_with_four(hand, kind_border_indices[i], kind_border_indices[i+1], puurIndices[i])
    

    return result


def calculate_trump_selection_score(cards, trump: int) -> int:
    # add your code here
    kind_border_indices = [0, 9, 18, 27, 36]
    score = 0
    lower_boundary_trump_kind_indices = kind_border_indices[trump]
    upper_boundary_trump_kind_indices = kind_border_indices[trump + 1]

    for i in range(len(cards)):
        if cards[i] >= lower_boundary_trump_kind_indices and cards[i] < upper_boundary_trump_kind_indices:
            score += trump_score[cards[i] % 9]
        else:
            score += no_trump_score[cards[i] % 9]   
    return score

def calculate_score_of_card(i, trump):
    ''' calculate the score of a card depending on the trump '''
    kind_border_indices = [0, 9, 18, 27, 36]
    score = 0
    lower_boundary_trump_kind_indices = kind_border_indices[trump]
    upper_boundary_trump_kind_indices = kind_border_indices[trump + 1]

    if i >= lower_boundary_trump_kind_indices and i < upper_boundary_trump_kind_indices:
        score += trump_score[i % 9]
    else:
        score += no_trump_score[i % 9]   
    return score