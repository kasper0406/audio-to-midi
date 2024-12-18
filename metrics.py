from datetime import datetime
from tensorboardX import SummaryWriter


def configure_tensorboard() -> SummaryWriter:
    now = datetime.now()
    train_time = now.isoformat(timespec='seconds')

    summary_writer = SummaryWriter(f"runs/{train_time}")
    return summary_writer
