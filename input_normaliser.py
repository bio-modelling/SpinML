import os
import paer
import progressbar as pb

directory='/Users/roberttoth/Desktop/aer/'

# Count aedat files for progress bar
files = 0
for filename in os.listdir(directory):
    if filename.endswith(".aedat"):
        files += 1

# Timer output format
widgets = ['Processing ' + str(files) + ' files:', pb.Percentage(), ' ',
           pb.Bar(marker='|'), ' ']

# Start timer
timer = pb.ProgressBar(widgets=widgets, maxval=files).start()
iter=0

for filename in os.listdir(directory):
    if filename.endswith(".aedat"):
        d = paer.aedata(paer.aefile(filename))
        d = d.filter_events('OFF')
        d = d.take_v2(89).change_timescale(length=200,start=0)
        d.save_to_mat(os.path.splitext(filename)[0]+'.mat')
        
        #Update timer
        iter += 1
        timer.update(iter)

# Stop timer
timer.finish()
