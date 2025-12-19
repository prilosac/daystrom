from copy import deepcopy

from daystrom.nodes.openrouter.chat import OpenRouterChat

OPENROUTER_KEY = ""


def main():
    node1 = OpenRouterChat(api_key=OPENROUTER_KEY)
    res = node1.invoke("This is a test message")
    node2 = OpenRouterChat(api_key=OPENROUTER_KEY, context=deepcopy(node1.context))
    res2 = node2.invoke("another test")
    print("----------------------------------------")
    print("----------------------------------------")
    print(node1.context)
    print("----------------------------------------")
    print(res)
    print("----------------------------------------")
    print("----------------------------------------")
    print("")
    print("----------------------------------------")
    print("----------------------------------------")
    print(node2.context)
    print("----------------------------------------")
    print(res2)
    print("----------------------------------------")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
