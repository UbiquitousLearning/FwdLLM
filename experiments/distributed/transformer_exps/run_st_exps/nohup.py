import argparse
import logging
import os
from time import sleep


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    return parser.parse_args()


def wait_for_the_training_process():
    pipe_path = "./tmp/fedml"
    if not os.path.exists(os.path.dirname(pipe_path)):
        try:
            os.makedirs(os.path.dirname(pipe_path))
        except OSError as exc:  # Guard against race condition
            print(exc)
    if not os.path.exists(pipe_path):
        open(pipe_path, 'w').close()
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Received: '%s'" % message)
                print("Training is finished. Start the next training with...")
                os.remove(pipe_path)
                return
            sleep(3)
            # print("Daemon is alive. Waiting for the training result.")


# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

os.system("kill $(ps aux | grep \"fedavg_main_st.py\" | grep -v grep | awk '{print $2}')")

PARTITION_METHOD="niid_label_clients=600_alpha=1"
USE_QUANTIZE=""
FREEZE_LAYERS="e,0,1,2,3,4,5"

os.system("mkdir ./tmp/; touch ./tmp/fedml")
os.system('nohup sh run_seq_tagging.sh '
            '{0} {1} {2} > ./tmp/onto_{0}_quantize={1}_freeze={2}.log 2>&1 &'.format(PARTITION_METHOD,USE_QUANTIZE,FREEZE_LAYERS))

wait_for_the_training_process()

logging.info("cleaning the training...")

# kill $(ps aux | grep fedavg_main_st.py | grep -v grep | awk '{print $2}')
os.system("kill $(ps aux | grep \"fedavg_main_st.py\" | grep -v grep | awk '{print $2}')")

