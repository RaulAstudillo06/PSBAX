

# Multiobjective
- DTLZ2 with noise
- ZDT2 with batch size = 5, 10

# Singleobjective
- Ackley
    - Batch size
        - [x] 10D with different batch sizes
    - Noise 
- Hartmann 
    - Batch size
        - [ ] 6D with different batch sizes
    - Noise


- Find process id
```
tmux ls  -F 'socket_path: #{socket_path} | session_name: #{session_name} | server_pid: #{pid} | pane_pid: #{pane_pid}'
```

- Terminate job
    - First ctrl + Z
```
kill -SIGTERM %1

# can check by 
pidof 1
```