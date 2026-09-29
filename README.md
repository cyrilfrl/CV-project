
# Phase 0

- [ ] Phase 1 videos
    - stable phone
    - 1 to 3 people pass in front of the camera as well as other moving objects
    - 1 indoor/1 outdoor
- [ ] Phase 2 videos
    - sideways oscillations left to right and right to left for 1 minute
    - no people on the video
    - 1 indoor/1 outdoor
- [ ] Phase 3 videos
    - sideways oscillations left to right and right to left for 1 minute
    - 1 to 3 people on the video as well as moving objects
    - 1 indoor/1 outdoor

# Phase 1

- [ ] 

# Protocol for Data Capture

1. hold phone still
    1.1 start video capture (static)
2. transfer files
3. push to github as lfs and submit
4. store files in /YYY-MM-DD

# Use
```bash
uv run main --max-frames 180 --step-frames 45 --video <mettre video dans /data puis nom.mp4 ici>
uv run main --max-frames 450 --step-frames 15
```

# TODO
- faster human segmentation process (maybe not NN)
- segmentation include objects around people (bag, ...)
- apply smoothing only where 0 from outside border to inside
- temporal diff. for movements detection
