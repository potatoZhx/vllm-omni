import re

unique = set()
with open('/home/mumura/omni/dataset/trace/cogvideox_trace.txt/cogvideox_trace.txt') as f:
    for line in f:
        m = re.search(r'height=(\d+), width=(\d+), num_frames=(\d+).*num_inference_steps=(\d+)', line)
        if m:
            unique.add(tuple(map(int, m.groups())))

with open('/home/mumura/omni/dataset/traits/cogvideox_trace_types.txt', 'w') as out:
    out.write(f'数量: {len(unique)}\n')
    out.write('height | width | num_frames | num_inference_steps\n')
    for h, w, nf, nis in sorted(unique):
        out.write(f'{h} | {w} | {nf} | {nis}\n')
