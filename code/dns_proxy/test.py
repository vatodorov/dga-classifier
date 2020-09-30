

from argparse import ArgumentParser



class Test():

    def __init__(self, appserver, appserverport):
        self.appserver = appserver
        self.appserverport = appserverport

    def print_arg(self):
        print(self.appserver, self.appserverport)


if __name__ == "__main__":
    parser = ArgumentParser(usage = "test.py [options]:\n")

    rungroup = parser.add_argument_group("Optional runtime parameters.")
    rungroup.add_argument("--appserver", help="The IP address of the server the scoring app is running on.")
    rungroup.add_argument("--appserverport", default="5001", help="The port the scoring app is running on.")

    options = parser.parse_args()

    # Get from the arguments the server IP and the port the scoring app is running on
    appserver = options.appserver
    appserverport = options.appserverport

    blah = Test(appserver, appserverport)


