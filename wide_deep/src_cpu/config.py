import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_info', type=str, default='',
    help='model info file.')
parser.add_argument(
    '--model_dir', type=str, default='',
    help='model dir.')
parser.add_argument(
    '--export_dir', type=str, default='',
    help='which dir to export savedmodel.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_and_deep'}.")
parser.add_argument(
    '--work_mode', type=str, default='train_and_eval',
    help="Valid work mode: {'train_and_eval', 'eval'}.")

parser.add_argument(
    '--train_epochs', type=int, default=1, help='Number of training epochs.')
parser.add_argument(
    '--max_steps', type=int, default=100000000, help='Number of training max steps.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=10000, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/tmp/census_data/adult.data',
    help='dir of the training data.')
parser.add_argument(
    '--test_data', type=str, default='/tmp/census_data/adult.test',
    help='dir of the test data.')

parser.add_argument(
    '--ps_hosts', type=str, default='',
    help='ps_hosts.')

parser.add_argument(
    '--worker_hosts', type=str, default='',
    help='worker_hosts.')

parser.add_argument(
    '--job_name', type=str, default='None',
    help='job name (chief | ps | worker).')

parser.add_argument(
    '--task_index', type=int, default='-1',
    help='task_index.')

FLAGS, unparsed = parser.parse_known_args()
