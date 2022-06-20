
from main import Detector
import argparse
import concurrent.futures


def call_fun(arg):
    print(arg[0], arg[1])
    detect_obj = Detector(arg[0], arg[1], "ssd", 0.4)
    detect_obj.start_stream()


def thread(arg: list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(call_fun, arg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False,
                        help='--input path of file / Link', nargs='*',type=str, default=["0"])
    parser.add_argument(
        "--output",
        help='--output port',
        required=False,
        nargs="*",
        type=str,
        default=["5555"],
    )
    args = parser.parse_args()
    input = args.input
    output = args.output
    if len(input) == len(output):
        arg = []
        for i in range (len(output)):
            print("input: %r" % input[i])
            print("output: %r" % output[i])
            arg.append([input[i], f"tcp://127.0.0.1:{output[i]}"])
        thread(arg)
