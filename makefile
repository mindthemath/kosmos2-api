dev:
	PORT=8020 MAX_BATCH_SIZE=8 NUM_API_SERVERS=2 LOG_LEVEL=INFO uv run --isolated --extra api server.py

fish.mov:
	curl -fsSL https://cdn.math.computer/v/kosmos2/fish/IMG_2867.MOV -o fish.mov

betty.mov:
	curl -fsSL https://cdn.math.computer/v/kosmos2/betty/IMG_2647.mov -o betty.mov

stripes.mov:
	curl -fsSL https://cdn.math.computer/v/kosmos2/stripes/input.mov -o stripes.mov

go.mp4:
	yt-dlp https://www.youtube.com/watch?v=LO2RPDZkY88

frames.tar.gz: out/frames/frame_000001.json
	cd out && tar cvzf frames.tar.gz frames/ && cd .. && mv out/frames.tar.gz .

input.mp4: frames/frame_000001.png
	ffmpeg -framerate 30 -i 'frames/frame_%06d.png' -c:v libx264 -crf 23 -pix_fmt yuv420p input.mp4
	# curl -fsSL https://cdn.math.computer/v/kosmos2/fish/sm/input.mp4 -o input.mp4

clean-frames:
	rm -rf frames && mkdir -p frames

scale=3
file=fish.mov
frames/frame_000001.png: $(file)
	mkdir -p frames
	ffmpeg -i $(file) -vf "fps=30,scale='min(iw/$(scale),iw):min(ih/$(scale),ih)'" frames/frame_%06d.png
	@du -sh frames

out/frame_000001.png: client.py bboxes.py frames/frame_000001.png
	rm -rf out && mkdir -p out
	uv run --isolated --extra viz --with ray client.py

output.mp4: out/frame_000001.png
	ffmpeg -framerate 30 -i 'out/frame_%06d.png' -c:v libx264 -crf 23 -pix_fmt yuv420p output.mp4

split: frames/frame_000001.png

stitch: output.mp4

movie: output.mp4

requirements.txt: pyproject.toml uv.lock
	uv pip compile pyproject.toml --extra api --extra viz --upgrade -o requirements.txt

requirements.api.txt: pyproject.toml uv.lock
	uv pip compile pyproject.toml --extra api --upgrade -o requirements.api.txt

build: requirements.api.txt
	docker build -t kosmos2-api:latest .

snowman.png:
	curl -fsSL https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png -o snowman.png

test: snowman.png
	curl -X POST -F "content=@snowman.png" http://127.0.0.1:8020/predict | jq .output

ptest: snowman.png
	seq 1 23 | parallel --jobs 24 "curl -X POST -F 'content=@snowman.png' http://127.0.0.1:8020/predict 2>&1 || echo 'Request failed'"

lint:
	uvx black .
	uvx isort --profile black .

run:
	docker run --rm -ti \
	--gpus all \
	-p 8020:8000 \
	-e NUM_API_SERVERS=$(or $(NUM_API_SERVERS),1) \
	-e MAX_BATCH_SIZE=$(or $(MAX_BATCH_SIZE),1) \
	-e LOG_LEVEL=$(or $(LOG_LEVEL),INFO) \
	-e PORT=8000 \
	mindthemath/kosmos2-api:latest

tag: build
	docker tag kosmos2-api:latest mindthemath/kosmos2-api:$$(date +%Y%m%d)-cu12.2.2
	docker tag kosmos2-api:latest mindthemath/kosmos2-api:$$(date +%Y%m%d)
	docker tag kosmos2-api:latest mindthemath/kosmos2-api:latest
	docker images | grep mindthemath/kosmos2-api

push: tag
	docker push mindthemath/kosmos2-api:$$(date +%Y%m%d)-cu12.2.2
	docker push mindthemath/kosmos2-api:$$(date +%Y%m%d)
	docker push mindthemath/kosmos2-api:latest

parse:
	@ls out/frames | parallel --jobs 12 jq .entities out/frames/{}

term=spider
ratio:
	@echo "scale=2; $$(make parse | grep -c $(term)) / $$(ls out/frames/ | wc -l)" | bc

upgrade:
	uv lock --upgrade
