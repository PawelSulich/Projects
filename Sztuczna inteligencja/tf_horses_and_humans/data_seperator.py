def data_seperation():
    i=0
    z ="horse-or-human/humans/"
    do ="test_data/humans/"
    import os
    from pathlib import Path
    for horse in os.listdir(z):
        if i>5:
            Path(z+horse).rename(do+horse)
            i=0
        i+=1