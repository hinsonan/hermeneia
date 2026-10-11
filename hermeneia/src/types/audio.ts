export interface WaveformPeaks {
  min_peaks: number[];
  max_peaks: number[];
  num_peaks: number;
  duration_seconds: number;
  channels: number;
  sample_rate: number;
}

export interface AudioFileState {
  filePath: string;
  fileName: string;
  peaks: WaveformPeaks | null;
  isLoading: boolean;
  error: string | null;
}

export interface TrimSelection {
  start: number;
  end: number;
}
