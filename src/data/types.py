import pickle
from io import BytesIO
from typing import List, Optional, Any, Union, Tuple

import av
import jax.numpy as jnp
import numpy as np
from av.video.frame import VideoFrame
from cvproc import h264_to_ndarrays
from pydantic import ConfigDict, BaseModel
from pydantic import field_serializer, model_validator

av.logging.set_level(av.logging.ERROR)


class Video(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    frames: List[np.ndarray]
    frame_count: int
    height: int
    width: int

    @classmethod
    def from_frames(cls, frames: List[np.ndarray]) -> 'Video':
        assert len(frames) > 0, 'Empty video'
        return cls.model_construct(
            frames=frames,
            frame_count=len(frames),
            height=frames[0].shape[0],
            width=frames[0].shape[1],
        )

    @classmethod
    def from_path(cls, source: Union[str, BytesIO]) -> 'Video':
        container = av.open(source)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.reformat(
                width=224,
                height=224,
                format='yuv420p',
                interpolation=av.video.reformatter.Interpolation.LANCZOS
            )
            frames.append(frame.to_ndarray(format='rgb24'))
        return cls.from_frames(frames)

    @field_serializer('frames')
    def ser_model(self, frames: List[np.ndarray]) -> bytes:
        print('Serializing video')
        buf = BytesIO()
        container = av.open(buf, mode='w', format='h264')
        stream = container.add_stream(
            'h264',
            options={
                'preset': 'ultrafast',
                'tune': 'fastdecode',
                'crf': '28',
            }
        )
        stream.pix_fmt = 'yuv420p'
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        for frame in frames:
            frame = VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return bytes(buf.getvalue())

    @model_validator(mode='before')
    @classmethod
    def validate_model(cls, data) -> Any:
        assert isinstance(data, dict), 'Only dict is supported'
        assert isinstance(data['frames'], bytes), 'Video is already decoded'
        try:
            frames = h264_to_ndarrays(data['frames'])
        except Exception as e:
            raise ValueError('Failed to decode video') from e
        assert len(frames) == data['frame_count'], 'Frame count mismatch: %d != %d' % (len(frames), data['frame_count'])
        data['frames'] = frames
        return data

    def sample_frames(self, n: int):
        if n > self.frame_count:
            last = [self.frames[-1]] * (n - self.frame_count)
            self.frames = self.frames + last
            self.frame_count = len(self.frames)
        indices = np.linspace(0, self.frame_count, n + 1, dtype=int)
        indices = np.random.randint(indices[:-1], indices[1:], n)
        self.frames = [self.frames[i] for i in indices]
        self.frame_count = len(self.frames)


class VideoSample(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    key: str
    cls: Union[int, str]
    video: Video
    extra: Optional[dict] = None

    @classmethod
    def from_bytes(cls, data: bytes) -> 'VideoSample':
        return VideoSample(**pickle.loads(data))

    def sample_frames(self, n) -> 'VideoSample':
        self.video.sample_frames(n)
        return self

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(int(self.cls)), np.asarray(self.video.frames, dtype=np.uint8).transpose(0, 3, 1, 2)

    def to_tensors(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cls, vid = self.to_arrays()
        cls = jnp.asarray(cls)
        vid = jnp.asarray(vid)
        return cls, vid
