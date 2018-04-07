STEP 1 : pip install tensorboardX

#### HOW TO WORK WITH TENSORBOARD IN LOCAL MACHINE ONLY ?

1. Run tensorboard_test.py
2. From the same directory(check if you see another folder runs(that is where all your tensorboard logs are saved)),
run the following command line argument <b>tensorboard --logfile runs</b>
3. This will give you a link which you can open on browser.


#### HOW TO VIEW TENSORBOARD ON LOCAL MACHINE WHILE IT RUNS ON A REMOTE MACHINE ?

1. Follow steps 1,2 and 3 as above on the remote machine except now open it on specific port <b>tensorboard --logfile runs --port 8889</b>.
2. Then on the local machine, ssh -N -L localhost:8888:\<tensorboard link on remote machine\> user@host_name
  
#### NOTE : Periodically keep deleting stuff from the runs file, the logs are heavy.
