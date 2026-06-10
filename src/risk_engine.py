def calculate_risk(emotion, behaviour):

    if emotion in ["Angry", "Fear"] and behaviour == "AGGRESSIVE":
        return "HIGH"

    elif emotion == "Sad" and behaviour == "AGGRESSIVE":
        return "HIGH"

    elif emotion == "Neutral" and behaviour == "NORMAL":
        return "LOW"

    elif emotion == "Happy" and behaviour == "SLOW":
        return "LOW"

    else:
        return "MEDIUM"


emotion = "Angry"
behaviour = "AGGRESSIVE"

risk = calculate_risk(emotion, behaviour)

print("Emotion:", emotion)
print("Behaviour:", behaviour)
print("Risk:", risk)