from io import BytesIO
from typing import List, Optional, Any

import av
from av.video.frame import VideoFrame
from cvproc import h264_to_ndarrays
from numpy import ndarray
from pydantic import ConfigDict, BaseModel
from pydantic import field_serializer, model_validator

av.logging.set_level(av.logging.ERROR)


class Video(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    frames: List[ndarray]
    frame_count: int
    height: int
    width: int

    @classmethod
    def from_frames(cls, frames: List[ndarray]) -> 'Video':
        return cls.model_construct(
            frames=frames,
            frame_count=len(frames),
            height=frames[0].shape[0],
            width=frames[0].shape[1],
        )

    @classmethod
    def from_path(cls, source: str | BytesIO) -> 'Video':
        container = av.open(source)
        return cls.from_frames([frame.to_rgb().to_ndarray() for frame in container.decode(video=0)])

    @field_serializer('frames')
    def ser_model(self, frames: List[ndarray]) -> bytes:
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
        return bytes(buf)

    @model_validator(mode='before')
    @classmethod
    def validate_model(cls, data) -> Any:
        assert isinstance(data, dict), 'Only dict is supported'
        assert isinstance(data['frames'], bytes), 'Video is already decoded'
        try:
            frames = h264_to_ndarrays(data['frames'])
        except Exception as e:
            raise ValueError('Failed to decode video') from e
        assert len(frames) == data['frame_count'], 'Frame count mismatch'
        data['frames'] = frames
        return data


class VideoSample(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    key: str
    cls: str
    video: Video
    extra: Optional[dict] = None
