TO run code for 72 run that has 24 without wind and 48 test with wind 
you have to back to code with commit massage "code that I run 48 with wind and 24 with out the wind"
and run python D3QN.py --multirun  enable_wind=true learning_rate=0.001 batch_size=256,64,16 done=true,false done_lag=true,false myAl=true,false  wind_power=10.0,15.0

to run 30 exparimnet to see effect of batch size and cos_freq go to commite 
python D3QN.py --multirun  enable_wind=ture batch_size=128,64,32,16,8 myAl=ture  wind_power=10.0 cos_freq=0.05,0.1,0.2,0.4,0.8