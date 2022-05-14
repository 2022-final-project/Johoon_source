from pyparsing import null_debug_action


mydic = {}
mydic["t1"] = 53
mydic["t2"] = "hi"

print(mydic["t1"])

if mydic["t2"] != "":
    print("hi dude")
