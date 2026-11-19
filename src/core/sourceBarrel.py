from importlib.util import find_spec


if find_spec("src.feedSrc") is None:
    FEEDS = []
else:
    from src.feedSrc.VietnameseNews import FEEDS as VF
    from src.feedSrc.ScientificPaper import FEEDS as SF
    from src.feedSrc.CryptoCurrentcy import FEEDS as CF
    from src.feedSrc.Geopolitical import FEEDS as GF
    from src.feedSrc.AI import FEEDS as AF
    from src.feedSrc.Agriculture import FEEDS as AGF
    from src.feedSrc.Mineral import FEEDS as MF

    FEEDS = VF + SF + CF + GF + AF + AGF + MF
