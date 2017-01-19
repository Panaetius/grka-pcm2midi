package ch.fhnw.tvver;

import java.io.File;
import java.io.IOException;
import java.util.EnumSet;
import java.util.SortedSet;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;

import ch.fhnw.ether.audio.IAudioRenderTarget;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.media.RenderProgram;

public abstract class AbstractPCM2MIDI {
	enum Flags {SYNTH, WAVE, REPORT, DEBUG, MAX_SPEED, DUMP_MIDI}

	private final PCM2MIDIShell p2ms;
	private       Throwable     exception;

	/**
	 * Signal a note on MIDI event. The note will be recorded at the frame time this method is called.
	 * @throws InvalidMidiDataException 
	 */
	protected void noteOn(int key, int velocity) {
		p2ms.noteOn(key, velocity);
	}

	/**
	 * Signal a note off MIDI event. The note will be recorded at the frame time this method is called.
	 * @throws InvalidMidiDataException 
	 */
	protected void noteOff(int key, int velocity) {
		p2ms.noteOn(key, 0);
	}

	/**
	 * Callback to initialize analysis pipeline. Add your render commands to <code>program</code>.
	 * 
	 * @param program The program which will be run for analysis.
	 */
	protected abstract void initializePipeline(RenderProgram<IAudioRenderTarget> program);
	
	/**
	 * Callback invoked at end of analysis. Add your cleanup / serialization etc. code here.
	 */
	protected void shutdown() {}

	/**
	 * Create a PCM2MIDI instance.
	 * 
	 * @param track The file to read samples from.
	 * @param flags Control output such as writing the reports and a WAV file.
	 * @throws UnsupportedAudioFileException Thrown if an exception occurred while reading the source file.
	 * @throws IOException Thrown if an exception occurred while reading the source file or writing one of the output files.
	 * @throws MidiUnavailableException Thrown when system MIDI synthesizer could not be opened.
	 * @throws InvalidMidiDataException 
	 * @throws RenderCommandException 
	 */
	protected AbstractPCM2MIDI(File track, EnumSet<Flags> flags) throws UnsupportedAudioFileException, IOException, MidiUnavailableException, InvalidMidiDataException, RenderCommandException {
		p2ms = new PCM2MIDIShell(track, flags);
	}

	//--- for testing

	protected final int[] getVelocities() {
		return p2ms.tracker.getVelocities();
	}
	
	protected MidiKeyTracker getKeyTracker() {
		return p2ms.tracker;
	}

	//--- internal interface

	final PCM2MIDIShell getShell() {
		return p2ms;
	}

	final SortedSet<MidiEvent> getRefMidi() {
		return p2ms.getRefMidi();
	}

	final void handleException(Throwable t) {
		this.exception = t;
	}

	final Throwable getException() {
		return exception;
	}

	final String getReport() {
		if(exception != null)
			return exception.getClass().getName() + ":" + exception.getMessage();
		else
			return p2ms.getReport();
	}

	final boolean getFlag(Flags flag) {
		return p2ms.getFlag(flag);
	}

	final void writeWAV(File file) throws IOException {
		try {
			p2ms.writeWAV(file);
		} catch(IOException ex) {
			throw ex;
		} catch(Throwable t) {
			throw new IOException(t);
		}
	}	
}
