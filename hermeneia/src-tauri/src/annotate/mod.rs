use crate::audio::{decode_audio_file_with_progress, prepare_speech_audio, DecodeProgressCallback};
use crate::error::{AudioError, Result};
use crate::runtime_cache::{global_runtime_cache, RuntimeCacheManager};
use crate::speaker::{
    diarize_prepared_audio_with_callbacks_cached, validate_diarize_params, DiarizationResult,
    DiarizeCallbacks, DiarizeParams, DiarizeStage, SpeakerDevice, SpeakerModel,
};
use crate::transcribe::{
    transcribe_prepared_audio_with_reporter_cached, ProgressReporter, TranscribeParams,
    TranscriptResult, TranscriptionTask, WhisperModel,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotateParams {
    pub transcribe: TranscribeParams,
    pub diarize: DiarizeParams,
    pub speaker_names: HashMap<i32, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedSegment {
    pub index: usize,
    pub start: f64,
    pub end: f64,
    pub speaker: i32,
    pub speaker_name: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedResult {
    pub segments: Vec<AnnotatedSegment>,
    pub speaker_names: HashMap<i32, String>,
    pub num_speakers: usize,
    pub language: Option<String>,
    pub audio_duration: f32,
    pub diarization_inference_time: f64,
    pub transcription_inference_time: f64,
    pub total_inference_time: f64,
    pub whisper_model: String,
    pub speaker_model: String,
    pub speaker_device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationPhase {
    Starting,
    DecodingAudio,
    PreparingAudio,
    LoadingSpeakerModel,
    EnsuringSpeakerModels,
    InitializingSpeakerRuntime,
    Diarizing,
    LoadingTranscriptionModel,
    Transcribing,
    Merging,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationProgress {
    pub phase: AnnotationPhase,
    pub current: Option<usize>,
    pub total: Option<usize>,
    pub message: String,
    pub indeterminate: bool,
}

impl AnnotationProgress {
    pub fn starting() -> Self {
        Self {
            phase: AnnotationPhase::Starting,
            current: None,
            total: None,
            message: "Starting annotation...".to_string(),
            indeterminate: true,
        }
    }

    pub fn decoding_audio() -> Self {
        Self {
            phase: AnnotationPhase::DecodingAudio,
            current: None,
            total: None,
            message: "Decoding audio...".to_string(),
            indeterminate: true,
        }
    }

    pub fn decoding_audio_progress(current: usize, total: usize) -> Self {
        if total > 0 {
            Self {
                phase: AnnotationPhase::DecodingAudio,
                current: Some(current),
                total: Some(total),
                message: "Decoding audio...".to_string(),
                indeterminate: false,
            }
        } else {
            Self::decoding_audio()
        }
    }

    pub fn preparing_audio() -> Self {
        Self {
            phase: AnnotationPhase::PreparingAudio,
            current: None,
            total: None,
            message: "Preparing mono 16kHz audio...".to_string(),
            indeterminate: true,
        }
    }

    pub fn loading_speaker_model() -> Self {
        Self {
            phase: AnnotationPhase::LoadingSpeakerModel,
            current: None,
            total: None,
            message: "Loading speaker diarization model...".to_string(),
            indeterminate: true,
        }
    }

    pub fn ensuring_speaker_models() -> Self {
        Self {
            phase: AnnotationPhase::EnsuringSpeakerModels,
            current: None,
            total: None,
            message: "Ensuring speaker diarization models...".to_string(),
            indeterminate: true,
        }
    }

    pub fn initializing_speaker_runtime() -> Self {
        Self {
            phase: AnnotationPhase::InitializingSpeakerRuntime,
            current: None,
            total: None,
            message: "Initializing speaker runtime...".to_string(),
            indeterminate: true,
        }
    }

    pub fn diarizing(current: usize, total: usize) -> Self {
        Self {
            phase: AnnotationPhase::Diarizing,
            current: Some(current),
            total: Some(total),
            message: "Diarizing...".to_string(),
            indeterminate: false,
        }
    }

    pub fn loading_transcription_model() -> Self {
        Self {
            phase: AnnotationPhase::LoadingTranscriptionModel,
            current: None,
            total: None,
            message: "Loading transcription model...".to_string(),
            indeterminate: true,
        }
    }

    pub fn transcribing(current: usize, total: usize) -> Self {
        Self {
            phase: AnnotationPhase::Transcribing,
            current: Some(current),
            total: Some(total),
            message: "Transcribing...".to_string(),
            indeterminate: false,
        }
    }

    pub fn merging() -> Self {
        Self {
            phase: AnnotationPhase::Merging,
            current: None,
            total: None,
            message: "Merging transcript with speaker segments...".to_string(),
            indeterminate: true,
        }
    }

    pub fn completed() -> Self {
        Self {
            phase: AnnotationPhase::Completed,
            current: Some(100),
            total: Some(100),
            message: "Annotation complete".to_string(),
            indeterminate: false,
        }
    }
}

pub trait AnnotationProgressReporter: Send + Sync {
    fn report(&self, progress: AnnotationProgress);
}

#[derive(Debug, Clone)]
pub struct NoAnnotationProgress;

impl AnnotationProgressReporter for NoAnnotationProgress {
    fn report(&self, _progress: AnnotationProgress) {}
}

pub fn parse_whisper_model(s: &str) -> Result<WhisperModel> {
    match s.to_lowercase().as_str() {
        "tiny" => Ok(WhisperModel::Tiny),
        "tiny.en" => Ok(WhisperModel::TinyEn),
        "base" => Ok(WhisperModel::Base),
        "base.en" => Ok(WhisperModel::BaseEn),
        "small" => Ok(WhisperModel::Small),
        "small.en" => Ok(WhisperModel::SmallEn),
        "medium" => Ok(WhisperModel::Medium),
        "medium.en" => Ok(WhisperModel::MediumEn),
        "large" => Ok(WhisperModel::Large),
        "large-v2" => Ok(WhisperModel::LargeV2),
        "large-v3" => Ok(WhisperModel::LargeV3),
        _ => Err(AudioError::InvalidTranscribeParams(format!(
            "Invalid transcribe model '{}'. Use: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, large-v2, large-v3",
            s
        ))),
    }
}

pub fn parse_speaker_model(s: &str) -> Result<SpeakerModel> {
    match s.to_lowercase().as_str() {
        "english" => Ok(SpeakerModel::English),
        "multilingual" => Ok(SpeakerModel::Multilingual),
        _ => Err(AudioError::InvalidDiarizeParams(format!(
            "Invalid speaker model '{}'. Use: english, multilingual",
            s
        ))),
    }
}

pub fn parse_speaker_device(s: &str) -> Result<SpeakerDevice> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(SpeakerDevice::Cpu),
        "cuda" => Ok(SpeakerDevice::Cuda),
        "coreml" => Ok(SpeakerDevice::CoreMl),
        _ => Err(AudioError::InvalidDiarizeParams(format!(
            "Invalid device '{}'. Use: cpu, cuda, coreml",
            s
        ))),
    }
}

pub fn parse_task(s: &str) -> Result<TranscriptionTask> {
    match s.to_lowercase().as_str() {
        "transcribe" => Ok(TranscriptionTask::Transcribe),
        "translate" => Ok(TranscriptionTask::Translate),
        _ => Err(AudioError::InvalidTranscribeParams(format!(
            "Invalid task '{}'. Use: transcribe, translate",
            s
        ))),
    }
}

pub fn parse_speaker_names(names_str: &str) -> HashMap<i32, String> {
    names_str
        .split(',')
        .enumerate()
        .filter_map(|(i, part)| {
            let (id, name) = match part.split_once('=') {
                Some((k, v)) => (k.trim().parse::<i32>().ok()?, v.trim()),
                None => (i as i32, part.trim()),
            };
            if name.is_empty() {
                None
            } else {
                Some((id, name.to_string()))
            }
        })
        .collect()
}

pub fn speaker_label(id: i32, names: &HashMap<i32, String>) -> String {
    names
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("Speaker {}", id))
}

struct TranscribeProgressAdapter {
    reporter: Arc<dyn AnnotationProgressReporter>,
}

impl ProgressReporter for TranscribeProgressAdapter {
    fn start(&self) {
        self.reporter
            .report(AnnotationProgress::loading_transcription_model());
    }

    fn report(&self, current: usize, total: usize) {
        self.reporter
            .report(AnnotationProgress::transcribing(current, total));
    }
}

fn check_cancelled(cancel_flag: &Option<Arc<AtomicBool>>) -> Result<()> {
    if cancel_flag
        .as_ref()
        .map(|flag| flag.load(Ordering::SeqCst))
        .unwrap_or(false)
    {
        return Err(AudioError::Cancelled);
    }
    Ok(())
}

pub fn annotate_audio_with_reporter(
    audio_path: &str,
    params: AnnotateParams,
    reporter: Arc<dyn AnnotationProgressReporter>,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<AnnotatedResult> {
    annotate_audio_with_reporter_cached(
        audio_path,
        params,
        reporter,
        cancel_flag,
        Some(global_runtime_cache()),
    )
}

pub fn annotate_audio_with_reporter_cached(
    audio_path: &str,
    params: AnnotateParams,
    reporter: Arc<dyn AnnotationProgressReporter>,
    cancel_flag: Option<Arc<AtomicBool>>,
    runtime_cache: Option<Arc<RuntimeCacheManager>>,
) -> Result<AnnotatedResult> {
    validate_diarize_params(&params.diarize)?;

    let start = Instant::now();
    if !params.transcribe.timestamps {
        tracing::warn!(
            "Annotation requested without timestamps; speaker assignment quality will be degraded."
        );
    }

    reporter.report(AnnotationProgress::starting());

    check_cancelled(&cancel_flag)?;

    reporter.report(AnnotationProgress::decoding_audio());
    tracing::info!("Decoding audio once for annotation: {}", audio_path);
    let reporter_for_decode = reporter.clone();
    let cancel_for_decode = cancel_flag.clone();
    let decode_progress: DecodeProgressCallback = Box::new(move |current, total| {
        reporter_for_decode.report(AnnotationProgress::decoding_audio_progress(current, total));
        !cancel_for_decode
            .as_ref()
            .map(|flag| flag.load(Ordering::SeqCst))
            .unwrap_or(false)
    });
    let audio = decode_audio_file_with_progress(audio_path, Some(decode_progress))?;
    check_cancelled(&cancel_flag)?;

    reporter.report(AnnotationProgress::preparing_audio());
    let speech_audio = prepare_speech_audio(&audio)?;
    drop(audio);
    check_cancelled(&cancel_flag)?;

    reporter.report(AnnotationProgress::loading_speaker_model());

    let reporter_for_diarize_stage = reporter.clone();
    let diarize_stage_progress = Arc::new(
        move |stage_progress: crate::speaker::DiarizeStageProgress| match stage_progress.stage {
            DiarizeStage::EnsuringModels => {
                reporter_for_diarize_stage.report(AnnotationProgress::ensuring_speaker_models());
            }
            DiarizeStage::InitializingRuntime => {
                reporter_for_diarize_stage
                    .report(AnnotationProgress::initializing_speaker_runtime());
            }
            DiarizeStage::Diarizing => {
                if let (Some(current), Some(total)) = (stage_progress.current, stage_progress.total)
                {
                    reporter_for_diarize_stage
                        .report(AnnotationProgress::diarizing(current, total));
                } else {
                    reporter_for_diarize_stage.report(AnnotationProgress {
                        phase: AnnotationPhase::Diarizing,
                        current: None,
                        total: None,
                        message: stage_progress.message.to_string(),
                        indeterminate: true,
                    });
                }
            }
            DiarizeStage::Finalizing => {
                reporter_for_diarize_stage.report(AnnotationProgress {
                    phase: AnnotationPhase::Diarizing,
                    current: None,
                    total: None,
                    message: stage_progress.message.to_string(),
                    indeterminate: true,
                });
            }
        },
    );

    let diarization = diarize_prepared_audio_with_callbacks_cached(
        &speech_audio,
        params.diarize.clone(),
        DiarizeCallbacks {
            chunk_progress: None,
            stage_progress: Some(diarize_stage_progress),
        },
        cancel_flag.clone(),
        runtime_cache.clone(),
    )?;

    check_cancelled(&cancel_flag)?;

    let adapter = TranscribeProgressAdapter {
        reporter: reporter.clone(),
    };

    // Ensure loading_model phase is emitted in annotate flow.
    adapter.start();

    let transcript = transcribe_prepared_audio_with_reporter_cached(
        &speech_audio,
        params.transcribe.clone(),
        &adapter,
        cancel_flag.clone(),
        runtime_cache,
    )?;

    check_cancelled(&cancel_flag)?;

    reporter.report(AnnotationProgress::merging());
    let result = merge_annotation_result(&transcript, &diarization, &params.speaker_names, &params);
    reporter.report(AnnotationProgress::completed());

    Ok(AnnotatedResult {
        total_inference_time: start.elapsed().as_secs_f64(),
        ..result
    })
}

pub fn annotate_audio(audio_path: &str, params: AnnotateParams) -> Result<AnnotatedResult> {
    annotate_audio_with_reporter_cached(
        audio_path,
        params,
        Arc::new(NoAnnotationProgress),
        None,
        Some(global_runtime_cache()),
    )
}

fn merge_annotation_result(
    transcript: &TranscriptResult,
    diarization: &DiarizationResult,
    overrides: &HashMap<i32, String>,
    params: &AnnotateParams,
) -> AnnotatedResult {
    let mut speaker_names = overrides.clone();

    for seg in &diarization.segments {
        speaker_names
            .entry(seg.speaker)
            .or_insert_with(|| format!("Speaker {}", seg.speaker));
    }

    let segments = transcript
        .segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let start = seg.start.unwrap_or(0.0);
            let end = seg.end.unwrap_or(start);
            let speaker = assign_speaker(start, end, diarization);
            AnnotatedSegment {
                index: i + 1,
                start,
                end,
                speaker,
                speaker_name: speaker_label(speaker, &speaker_names),
                text: seg.text.trim().to_string(),
            }
        })
        .collect();

    AnnotatedResult {
        segments,
        speaker_names,
        num_speakers: diarization.num_speakers,
        language: transcript.language.clone(),
        audio_duration: diarization.audio_duration,
        diarization_inference_time: diarization.inference_time,
        transcription_inference_time: transcript.inference_time,
        total_inference_time: 0.0,
        whisper_model: format!("{:?}", params.transcribe.model),
        speaker_model: params.diarize.model.display_name().to_string(),
        speaker_device: params.diarize.device.provider_string().to_string(),
    }
}

fn assign_speaker(seg_start: f64, seg_end: f64, diarization: &DiarizationResult) -> i32 {
    if diarization.segments.is_empty() {
        return 0;
    }

    let mut best_overlap = 0.0f64;
    let mut overlap_speaker: Option<i32> = None;

    for s in &diarization.segments {
        let start = s.start as f64;
        let end = s.end as f64;
        let overlap = (f64::min(seg_end, end) - f64::max(seg_start, start)).max(0.0);

        if overlap > best_overlap {
            best_overlap = overlap;
            overlap_speaker = Some(s.speaker);
        }
    }

    if let Some(id) = overlap_speaker {
        return id;
    }

    let midpoint = if seg_end > seg_start {
        (seg_start + seg_end) / 2.0
    } else {
        seg_start
    };

    diarization
        .segments
        .iter()
        .map(|s| {
            let center = (s.start as f64 + s.end as f64) / 2.0;
            let distance = (center - midpoint).abs();
            (s.speaker, distance)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| id)
        .unwrap_or(0)
}

pub fn format_as_srt(result: &AnnotatedResult) -> String {
    result
        .segments
        .iter()
        .map(|seg| {
            format!(
                "{}\n{} --> {}\n[{}] {}\n",
                seg.index,
                format_timestamp(seg.start),
                format_timestamp(seg.end),
                seg.speaker_name,
                seg.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_as_text(result: &AnnotatedResult) -> String {
    let mut out = String::new();
    for seg in &result.segments {
        let start_min = (seg.start / 60.0).floor() as u32;
        let start_sec = (seg.start % 60.0).floor() as u32;
        out.push_str(&format!(
            "[{:02}:{:02}] {}: {}\n",
            start_min, start_sec, seg.speaker_name, seg.text
        ));
    }
    out
}

pub fn format_as_json(result: &AnnotatedResult) -> Result<String> {
    serde_json::to_string_pretty(result)
        .map_err(|e| AudioError::TranscriptionFailed(format!("JSON serialization failed: {}", e)))
}

fn format_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0).floor() as u32;
    let minutes = ((seconds % 3600.0) / 60.0).floor() as u32;
    let secs = (seconds % 60.0).floor() as u32;
    let millis = ((seconds % 1.0) * 1000.0).floor() as u32;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speaker::SpeakerSegment;
    use crate::transcribe::TranscriptSegment;

    fn mk_diarization(segments: Vec<SpeakerSegment>) -> DiarizationResult {
        DiarizationResult {
            segments,
            num_speakers: 2,
            audio_duration: 10.0,
            inference_time: 1.0,
            model: "test".to_string(),
            device: "cpu".to_string(),
        }
    }

    #[test]
    fn test_assign_speaker_overlap() {
        let diar = mk_diarization(vec![
            SpeakerSegment {
                speaker: 0,
                start: 0.0,
                end: 2.0,
            },
            SpeakerSegment {
                speaker: 1,
                start: 2.0,
                end: 5.0,
            },
        ]);
        let spk = assign_speaker(2.2, 3.0, &diar);
        assert_eq!(spk, 1);
    }

    #[test]
    fn test_assign_speaker_no_overlap_uses_nearest_midpoint() {
        let diar = mk_diarization(vec![
            SpeakerSegment {
                speaker: 0,
                start: 10.0,
                end: 12.0,
            },
            SpeakerSegment {
                speaker: 1,
                start: 30.0,
                end: 31.0,
            },
        ]);
        let spk = assign_speaker(25.0, 25.0, &diar);
        assert_eq!(spk, 1);
    }

    #[test]
    fn test_assign_speaker_empty_diarization_defaults_zero() {
        let diar = mk_diarization(vec![]);
        let spk = assign_speaker(0.0, 1.0, &diar);
        assert_eq!(spk, 0);
    }

    #[test]
    fn test_json_end_falls_back_to_start() {
        let mut names = HashMap::new();
        names.insert(0, "Alice".to_string());
        let params = AnnotateParams {
            transcribe: TranscribeParams::default(),
            diarize: DiarizeParams::default(),
            speaker_names: names,
        };
        let transcript = TranscriptResult {
            segments: vec![TranscriptSegment {
                id: 0,
                start: Some(1.5),
                end: None,
                text: "hello".to_string(),
            }],
            text: "hello".to_string(),
            language: Some("en".to_string()),
            duration: 1.5,
            model: WhisperModel::Tiny,
            inference_time: 1.0,
        };
        let diar = mk_diarization(vec![SpeakerSegment {
            speaker: 0,
            start: 0.0,
            end: 2.0,
        }]);

        let result = merge_annotation_result(&transcript, &diar, &HashMap::new(), &params);
        assert_eq!(result.segments[0].start, result.segments[0].end);
    }

    #[test]
    fn test_validate_diarize_params_failures() {
        let invalid_threshold = DiarizeParams {
            threshold: 2.0,
            ..DiarizeParams::default()
        };
        assert!(validate_diarize_params(&invalid_threshold).is_err());

        let invalid_speakers = DiarizeParams {
            num_speakers: Some(0),
            ..DiarizeParams::default()
        };
        assert!(validate_diarize_params(&invalid_speakers).is_err());
    }
}
