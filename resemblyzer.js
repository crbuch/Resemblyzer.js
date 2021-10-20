
/*
the functions for computing the mel spectrograms came from fft.js 
(https://github.com/indutny/fft.js)
and magenta.js
(https://github.com/magenta/magenta-js)


Input for melSpectrogram()
 {
  sampleRate: number;
  hopLength: number;
  winLength: number;
  nFft: number;
  nMels: number;
  power: number;
  fMin: number;
  fMax: number;
}
*/

function hannWindow(length) {
	const win = new Float32Array(length);
	for (let i = 0; i < length; i++) {
		win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
	}
	return win;
}

function padCenterToLength(data, length) {
	// If data is longer than length, error!
	if (data.length > length) {
		throw new Error('Data is longer than length.');
	}

	const paddingLeft = Math.floor((length - data.length) / 2);
	const paddingRight = length - data.length - paddingLeft;
	return padConstant(data, [paddingLeft, paddingRight]);
}


function padConstant(data, padding) {
	let padLeft, padRight;
	if (typeof padding === 'object') {
		[padLeft, padRight] = padding;
	} else {
		padLeft = padRight = padding;
	}
	const out = new Float32Array(data.length + padLeft + padRight);
	out.set(data, padLeft);
	return out;
}


function padReflect(data, padding) {
	const out = padConstant(data, padding);
	for (let i = 0; i < padding; i++) {
		// Pad the beginning with reflected values.
		out[i] = out[2 * padding - i];
		// Pad the end with reflected values.
		out[out.length - i - 1] = out[out.length - 2 * padding + i - 1];
	}
	return out;
}



function frame(
	data, frameLength,
	hopLength) {
	const bufferCount = Math.floor((data.length - frameLength) / hopLength) + 1;
	const buffers = Array.from(
		{ length: bufferCount }, (x, i) => new Float32Array(frameLength));
	for (let i = 0; i < bufferCount; i++) {
		const ind = i * hopLength;
		const buffer = data.slice(ind, ind + frameLength);
		buffers[i].set(buffer);
		// In the end, we will likely have an incomplete buffer, which we should
		// just ignore.
		if (buffer.length !== frameLength) {
			continue;
		}
	}
	return buffers;
}




function applyWindow(buffer, win) {
	if (buffer.length !== win.length) {
		console.error(
			`Buffer length ${buffer.length} != window length ${win.length}.`);
		return null;
	}

	const out = new Float32Array(buffer.length);
	for (let i = 0; i < buffer.length; i++) {
		out[i] = win[i] * buffer[i];
	}
	return out;
}



function FFT(size) {
	this.size = size | 0;
	if (this.size <= 1 || (this.size & (this.size - 1)) !== 0)
		throw new Error('FFT size must be a power of two and bigger than 1');

	this._csize = size << 1;

	// NOTE: Use of `var` is intentional for old V8 versions
	var table = new Array(this.size * 2);
	for (var i = 0; i < table.length; i += 2) {
		const angle = Math.PI * i / this.size;
		table[i] = Math.cos(angle);
		table[i + 1] = -Math.sin(angle);
	}
	this.table = table;

	// Find size's power of two
	var power = 0;
	for (var t = 1; this.size > t; t <<= 1)
		power++;

	// Calculate initial step's width:
	//   * If we are full radix-4 - it is 2x smaller to give inital len=8
	//   * Otherwise it is the same as `power` to give len=4
	this._width = power % 2 === 0 ? power - 1 : power;

	// Pre-compute bit-reversal patterns
	this._bitrev = new Array(1 << this._width);
	for (var j = 0; j < this._bitrev.length; j++) {
		this._bitrev[j] = 0;
		for (var shift = 0; shift < this._width; shift += 2) {
			var revShift = this._width - shift - 2;
			this._bitrev[j] |= ((j >>> shift) & 3) << revShift;
		}
	}

	this._out = null;
	this._data = null;
	this._inv = 0;
}



FFT.prototype.createComplexArray = function createComplexArray() {
	const res = new Array(this._csize);
	for (var i = 0; i < res.length; i++)
		res[i] = 0;
	return res;
};


FFT.prototype.toComplexArray = function toComplexArray(input, storage) {
	var res = storage || this.createComplexArray();
	for (var i = 0; i < res.length; i += 2) {
		res[i] = input[i >>> 1];
		res[i + 1] = 0;
	}
	return res;
};



FFT.prototype._singleTransform4 = function _singleTransform4(outOff, off,
	step) {
	const out = this._out;
	const data = this._data;
	const inv = this._inv ? -1 : 1;
	const step2 = step * 2;
	const step3 = step * 3;

	// Original values
	const Ar = data[off];
	const Ai = data[off + 1];
	const Br = data[off + step];
	const Bi = data[off + step + 1];
	const Cr = data[off + step2];
	const Ci = data[off + step2 + 1];
	const Dr = data[off + step3];
	const Di = data[off + step3 + 1];

	// Pre-Final values
	const T0r = Ar + Cr;
	const T0i = Ai + Ci;
	const T1r = Ar - Cr;
	const T1i = Ai - Ci;
	const T2r = Br + Dr;
	const T2i = Bi + Di;
	const T3r = inv * (Br - Dr);
	const T3i = inv * (Bi - Di);

	// Final values
	const FAr = T0r + T2r;
	const FAi = T0i + T2i;

	const FBr = T1r + T3i;
	const FBi = T1i - T3r;

	const FCr = T0r - T2r;
	const FCi = T0i - T2i;

	const FDr = T1r - T3i;
	const FDi = T1i + T3r;

	out[outOff] = FAr;
	out[outOff + 1] = FAi;
	out[outOff + 2] = FBr;
	out[outOff + 3] = FBi;
	out[outOff + 4] = FCr;
	out[outOff + 5] = FCi;
	out[outOff + 6] = FDr;
	out[outOff + 7] = FDi;
};


FFT.prototype._singleTransform2 = function _singleTransform2(outOff, off,
	step) {
	const out = this._out;
	const data = this._data;

	const evenR = data[off];
	const evenI = data[off + 1];
	const oddR = data[off + step];
	const oddI = data[off + step + 1];

	const leftR = evenR + oddR;
	const leftI = evenI + oddI;
	const rightR = evenR - oddR;
	const rightI = evenI - oddI;

	out[outOff] = leftR;
	out[outOff + 1] = leftI;
	out[outOff + 2] = rightR;
	out[outOff + 3] = rightI;
};



FFT.prototype._transform4 = function _transform4() {
	var out = this._out;
	var size = this._csize;

	// Initial step (permute and transform)
	var width = this._width;
	var step = 1 << width;
	var len = (size / step) << 1;

	var outOff;
	var t;
	var bitrev = this._bitrev;
	if (len === 4) {
		for (outOff = 0, t = 0; outOff < size; outOff += len, t++) {
			const off = bitrev[t];
			this._singleTransform2(outOff, off, step);
		}
	} else {
		// len === 8
		for (outOff = 0, t = 0; outOff < size; outOff += len, t++) {
			const off = bitrev[t];
			this._singleTransform4(outOff, off, step);
		}
	}

	// Loop through steps in decreasing order
	var inv = this._inv ? -1 : 1;
	var table = this.table;
	for (step >>= 2; step >= 2; step >>= 2) {
		len = (size / step) << 1;
		var quarterLen = len >>> 2;

		// Loop through offsets in the data
		for (outOff = 0; outOff < size; outOff += len) {
			// Full case
			var limit = outOff + quarterLen;
			for (var i = outOff, k = 0; i < limit; i += 2, k += step) {
				const A = i;
				const B = A + quarterLen;
				const C = B + quarterLen;
				const D = C + quarterLen;

				// Original values
				const Ar = out[A];
				const Ai = out[A + 1];
				const Br = out[B];
				const Bi = out[B + 1];
				const Cr = out[C];
				const Ci = out[C + 1];
				const Dr = out[D];
				const Di = out[D + 1];

				// Middle values
				const MAr = Ar;
				const MAi = Ai;

				const tableBr = table[k];
				const tableBi = inv * table[k + 1];
				const MBr = Br * tableBr - Bi * tableBi;
				const MBi = Br * tableBi + Bi * tableBr;

				const tableCr = table[2 * k];
				const tableCi = inv * table[2 * k + 1];
				const MCr = Cr * tableCr - Ci * tableCi;
				const MCi = Cr * tableCi + Ci * tableCr;

				const tableDr = table[3 * k];
				const tableDi = inv * table[3 * k + 1];
				const MDr = Dr * tableDr - Di * tableDi;
				const MDi = Dr * tableDi + Di * tableDr;

				// Pre-Final values
				const T0r = MAr + MCr;
				const T0i = MAi + MCi;
				const T1r = MAr - MCr;
				const T1i = MAi - MCi;
				const T2r = MBr + MDr;
				const T2i = MBi + MDi;
				const T3r = inv * (MBr - MDr);
				const T3i = inv * (MBi - MDi);

				// Final values
				const FAr = T0r + T2r;
				const FAi = T0i + T2i;

				const FCr = T0r - T2r;
				const FCi = T0i - T2i;

				const FBr = T1r + T3i;
				const FBi = T1i - T3r;

				const FDr = T1r - T3i;
				const FDi = T1i + T3r;

				out[A] = FAr;
				out[A + 1] = FAi;
				out[B] = FBr;
				out[B + 1] = FBi;
				out[C] = FCr;
				out[C + 1] = FCi;
				out[D] = FDr;
				out[D + 1] = FDi;
			}
		}
	}
};


FFT.prototype.transform = function transform(out, data) {
	if (out === data)
		throw new Error('Input and output buffers must be different');

	this._out = out;
	this._data = data;
	this._inv = 0;
	this._transform4();
	this._out = null;
	this._data = null;
};



function fft(y) {
	const fft = new FFT(y.length);
	const out = fft.createComplexArray();
	const data = fft.toComplexArray(y);
	fft.transform(out, data);
	return out;
}



function stft(y, params) {
	const nFft = params.nFft || 2048;
	const winLength = params.winLength || nFft;
	const hopLength = params.hopLength || Math.floor(winLength / 4);

	let fftWindow = hannWindow(winLength);

	// Pad the window to be the size of nFft.
	fftWindow = padCenterToLength(fftWindow, nFft);

	// Pad the time series so that the frames are centered.
	y = padReflect(y, Math.floor(nFft / 2));

	// Window the time series.
	const yFrames = frame(y, nFft, hopLength);
	// Pre-allocate the STFT matrix.
	const stftMatrix = [];

	const width = yFrames.length;
	const height = nFft + 2;
	for (let i = 0; i < width; i++) {
		// Each column is a Float32Array of size height.
		const col = new Float32Array(height);
		stftMatrix[i] = col;
	}

	for (let i = 0; i < width; i++) {
		// Populate the STFT matrix.
		const winBuffer = applyWindow(yFrames[i], fftWindow);
		const col = fft(winBuffer);
		stftMatrix[i].set(col.slice(0, height));
	}

	return stftMatrix;
}


function pow(arr, power) {
	return arr.map((v) => Math.pow(v, power));
}

function mag(y) {
	const out = new Float32Array(y.length / 2);
	for (let i = 0; i < y.length / 2; i++) {
		out[i] = Math.sqrt(y[i * 2] * y[i * 2] + y[i * 2 + 1] * y[i * 2 + 1]);
	}
	return out;
}

function magSpectrogram(stft, power) {
	const spec = stft.map((fft) => pow(mag(fft), power));
	const nFft = stft[0].length - 1;
	return [spec, nFft];
}




function calculateFftFreqs(sampleRate, nFft) {
	return linearSpace(0, sampleRate / 2, Math.floor(1 + nFft / 2));
}


function hzToMel(hz) {
	return 1125.0 * Math.log(1 + hz / 700.0);
}


function linearSpace(start, end, count) {
	// Include start and endpoints.
	const delta = (end - start) / (count - 1);
	const out = new Float32Array(count);
	for (let i = 0; i < count; i++) {
		out[i] = start + delta * i;
	}
	return out;
}


function melToHz(mel) {
	return 700.0 * (Math.exp(mel / 1125.0) - 1);
}



function calculateMelFreqs(
	nMels, fMin, fMax) {
	const melMin = hzToMel(fMin);
	const melMax = hzToMel(fMax);

	// Construct linearly spaced array of nMel intervals, between melMin and
	// melMax.
	const mels = linearSpace(melMin, melMax, nMels);
	const hzs = mels.map((mel) => melToHz(mel));
	return hzs;
}



function internalDiff(arr) {
	const out = new Float32Array(arr.length - 1);
	for (let i = 0; i < arr.length; i++) {
		out[i] = arr[i + 1] - arr[i];
	}
	return out;
}


function outerSubtract(arr, arr2) {
	const out = [];
	for (let i = 0; i < arr.length; i++) {
		out[i] = new Float32Array(arr2.length);
	}
	for (let i = 0; i < arr.length; i++) {
		for (let j = 0; j < arr2.length; j++) {
			out[i][j] = arr[i] - arr2[j];
		}
	}
	return out;
}


function createMelFilterbank(params) {
	const fMin = params.fMin || 0;
	const fMax = params.fMax || params.sampleRate / 2;
	const nMels = params.nMels || 128;
	const nFft = params.nFft || 2048;

	// Center freqs of each FFT band.
	const fftFreqs = calculateFftFreqs(params.sampleRate, nFft);
	// (Pseudo) center freqs of each Mel band.
	const melFreqs = calculateMelFreqs(nMels + 2, fMin, fMax);

	const melDiff = internalDiff(melFreqs);
	const ramps = outerSubtract(melFreqs, fftFreqs);
	const filterSize = ramps[0].length;

	const weights = [];
	for (let i = 0; i < nMels; i++) {
		weights[i] = new Float32Array(filterSize);
		for (let j = 0; j < ramps[i].length; j++) {
			const lower = -ramps[i][j] / melDiff[i];
			const upper = ramps[i + 2][j] / melDiff[i + 1];
			const weight = Math.max(0, Math.min(lower, upper));
			weights[i][j] = weight;
		}
	}

	// Slaney-style mel is scaled to be approx constant energy per channel.
	for (let i = 0; i < weights.length; i++) {
		// How much energy per channel.
		const enorm = 2.0 / (melFreqs[2 + i] - melFreqs[i]);
		// Normalize by that amount.
		weights[i] = weights[i].map((val) => val * enorm);
	}

	return weights;
}

function applyFilterbank(
	mags, filterbank) {
	if (mags.length !== filterbank[0].length) {
		throw new Error(
			`Each entry in filterbank should have dimensions ` +
			`matching FFT. |mags| = ${mags.length}, ` +
			`|filterbank[0]| = ${filterbank[0].length}.`);
	}

	// Apply each filter to the whole FFT signal to get one value.
	const out = new Float32Array(filterbank.length);
	for (let i = 0; i < filterbank.length; i++) {
		// To calculate filterbank energies we multiply each filterbank with the
		// power spectrum.
		const win = applyWindow(mags, filterbank[i]);
		// Then add up the coefficents.
		out[i] = win.reduce((a, b) => a + b);
	}
	return out;
}

function applyWholeFilterbank(spec, filterbank) {
	// Apply a point-wise dot product between the array of arrays.
	const out = [];
	for (let i = 0; i < spec.length; i++) {
		out[i] = applyFilterbank(spec[i], filterbank);
	}
	return out;
}


var melSpectrogram = function (y, params) {
	if (!params.power) {
		params.power = 2.0;
	}
	const stftMatrix = stft(y, params);
	const [spec, nFft] = magSpectrogram(stftMatrix, params.power);

	params.nFft = nFft;
	const melBasis = createMelFilterbank(params);
	return applyWholeFilterbank(spec, melBasis);
}


//functions to compute the partial slices of samples and get spectrograms


function compute_partial_slices(n_samples, rate, min_coverage) {
	//taken from voice_encoder.py from Resemblyzer
	let samples_per_frame = parseInt((16000 * 10 / 1000));
	let n_frames = parseInt(Math.ceil((n_samples + 1) / samples_per_frame));
	let frame_step = parseInt(Math.round((16000 / rate) / samples_per_frame));
	let wav_slices = [];
	let mel_slices = [];
	let steps = Math.max(1, n_frames - 160 + frame_step + 1);
	for (let i = 0; i < steps; i += frame_step) {
		mel_range = [i, i + 160];
		wav_range = [];
		for (let j = 0; j < mel_range.length; j++) {
			wav_range.push(mel_range[j] * samples_per_frame);
		}
		mel_slices.push(mel_range);
		wav_slices.push(wav_range);
	}
	let last_wav_range = wav_slices[wav_slices.length - 1];
	let coverage = (n_samples - last_wav_range[0]) / (last_wav_range[1] - last_wav_range[0]);
	if (coverage < min_coverage && mel_slices.length > 1) {
		mel_slices = mel_slices.slice(0, mel_slices.length - 1);
		wav_slices = wav_slices.slice(0, wav_slices.length - 1);
	}
	return [wav_slices, mel_slices];
}

function wav_to_mel_spectrogram(wav) {
	return melSpectrogram(wav, { sampleRate: 16000, hopLength: Math.round(16000 * 10 / 1000), winLength: 0, nFft: 512, nMels: 40, power: 2.0, fMin: 0.0, fMax: 8000 })
}

var get_partial_mels = function(wav, rate = 1.3, min_coverage = 0.75, model) {
	//taken from voice_encoder.py from Resemblyzer
	let tmpslc = compute_partial_slices(wav.length, rate, min_coverage);
	let wav_slices = tmpslc[0]
	let mel_slices = tmpslc[1]


	max_wave_length = wav_slices[wav_slices.length - 1][1];
	if (max_wave_length >= wav.length) {
		wav = Array.from(wav)
		const stop = max_wave_length - wav.length;
		for (let i = 0; i < stop; i++) {
			wav.push(0)
		}
	}
	let mel = wav_to_mel_spectrogram(wav);
	let mels = [];


	for (let i = 0; i < mel_slices.length; i++) {
		mels.push(mel.slice(mel_slices[i][0], mel_slices[i][1]));
	}
	//returns partial slices of mels (160 frames per mel with 40 channels)
	return mels;
}



//these next functions were written by me

//use webkitAudioContext if browser is safari
window.AudioContext = window.AudioContext || window.webkitAudioContext;
const audioContext = new AudioContext();
let currentBuffer = null;


const resample = function (source, target) {
	//resamples audio to a sampling rate of 16000
	let newBuffer;
	let resolved = false;
	let promise = new Promise(function (resolve) {
		let intv3 = setInterval(function () {
			if (resolved == true) {
				resolve(newBuffer);
				clearInterval(intv3);
			}
		})
	})
	var TARGET_SAMPLE_RATE = target;
	var offlineCtx = new OfflineAudioContext(source.numberOfChannels,
		source.duration * TARGET_SAMPLE_RATE,
		TARGET_SAMPLE_RATE);
	var offlineSource = offlineCtx.createBufferSource();
	offlineSource.buffer = source;
	offlineSource.connect(offlineCtx.destination);
	offlineSource.start();
	offlineCtx.startRendering().then((resampled) => {
		newBuffer = resampled;
		resolved = true;
	});
	return promise;
}
const arrayAverage = function (a, b) {
	//converts stereo to mono by averaging the corresponding samples in each channel
	let na = [];
	const stop = a.length;
	for (let i = 0; i < stop; i++) {
		na.push((a[i] + b[i]) / 2)
	}
	return na;
}
const getSamples = function (url) {
	//retrieve samples from a url
	let resolved = false;
	let samps = [];
	let promise = new Promise(function (resolve) {
		let intv1 = setInterval(function () {
			if (resolved == true) {
				resolve(samps);
				clearInterval(intv1);
			}
		})
	})
	if (url !== "") {
		fetch(url)
			.then(response => response.arrayBuffer())
			.then(arrayBuffer => audioContext.decodeAudioData(arrayBuffer))
			.then(audioBuffer => async function (buffer) {
				let asMono = [16000];
				buffer = await resample(buffer, 16000)
				samps.push(buffer.sampleRate)
				for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
					samps.push(buffer.getChannelData(i))
				}
				asMono.push(samps[1])
				for (let i = 2; i < samps.length; i++) {
					asMono[1] = arrayAverage(asMono[1], samps[i])
				}
				samps = asMono;
				resolved = true;
			}(audioBuffer));
	} else {
		resolved = true;
	}
	return promise;
}
const trim_silence = function (data) {
	//gets rid of unneccessary silence at beginning and end of file.
	while (data[1][0] == 0) {
		data[1].shift();
	}
	while (data[1].at(-1) == 0) {
		data[1].pop();
	}
	return data;
}
const normalize = function (data) {
	//normalizes audio to 0db
	let greatest = 0;
	const samps = data[1];
	const stop = samps.length;
	for (let i = 0; i < stop; i++) {
		const num = Math.abs(samps[i])
		if (num > greatest) {
			greatest = num;
		}
	}
	const multiplier = 1 / greatest
	for (let i = 0; i < stop; i++) {
		samps[i] = samps[i] * multiplier;
	}
	return data;
}
const preprocess_wav = function (data) {
	//wrapper for those 2 functions
	return normalize(trim_silence(data))
}


//creates a dummy varaible for the ort session
var __sess;

const infer = async function (mel) {
	//infers the embeddings from a partial mel
	let t = tf.tensor(mel)
	t = tf.reshape(t, [6400])
	let input = new ort.Tensor(t.dataSync(), [1, 160, 40])
	let output = await __sess.run({ "input.1": input })
	return output["119"].data;
}


var initialize_model = async function () {
	//starts the session if session is undefined
	__sess = await ort.InferenceSession.create("pretrained.onnx")
}

var embed_audio = async function (url) {
	//wrapper for everything here
	if (__sess == undefined) {
		//if the session wasn't started, start it
		console.log("Session starting, this may take a second.");
		await initialize_model();
	}

	//preprocess audio and get the mels
	let samps = await getSamples(url);
	samps = preprocess_wav(samps)[1];
	const mels = get_partial_mels(samps);

	//infer partial embeddings and add them to embeddings array
	let embeddings = [];
	const melstop = mels.length
	for (let i = 0; i < melstop; i++) {
		embeddings.push((await infer(mels[i])));
	}

	//gets the average of all embeddings in embedding array along an axis of 0
	let single_embedding = tf.mean(tf.tensor(embeddings), axis = 0);

	//convert the tensor to an array to normalize the embeddings
	let single_embedding_array = single_embedding.arraySync();
	const dividor = tf.norm(single_embedding, 2).arraySync();

	const stop1 = single_embedding_array.length
	for (let i = 0; i < stop1; i++) {
		single_embedding_array[i] /= dividor;
	}
	//convert it back to a tensor if needed for other models
	return tf.tensor(single_embedding_array)
}
