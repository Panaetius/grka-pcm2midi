package ch.fhnw.tvver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.ShortMessage;
import javax.sound.sampled.UnsupportedAudioFileException;
import javax.swing.JFrame;
import javax.swing.JLabel;

import ch.fhnw.ether.audio.IAudioRenderTarget;
import ch.fhnw.ether.audio.fx.AutoGain;
import ch.fhnw.ether.media.AbstractRenderCommand;
import ch.fhnw.ether.media.IScheduler;
import ch.fhnw.ether.media.Parameter;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.media.RenderProgram;

/**
 * A fake PCM2MIDI implementation which jitters the reference notes
 * and signals the jittered reference notes.
 *
 * @author simon.schubiger@fhnw.ch
 *
 */
public class DataGen extends AbstractPCM2MIDI {
	public DataGen(File track) throws UnsupportedAudioFileException, IOException, MidiUnavailableException, InvalidMidiDataException, RenderCommandException {
		super(track, EnumSet.of(Flags.REPORT, Flags.WAVE));
	}

	@Override
	protected void initializePipeline(RenderProgram<IAudioRenderTarget> program) {
		program.addLast(new AutoGain());
		program.addLast(new PCM2MIDI());

		JFrame frame = new JFrame();
		frame.setVisible(true);
		frame.add(new JLabel("Hello World"));
	}

	public class PCM2MIDI extends AbstractRenderCommand<IAudioRenderTarget> {
		private final List<List<MidiEvent>> midiRef = new ArrayList<>();
		private       int                   msTime;

		private HashSet<int> currentMidiRef = new HashSet<>();

		private List<float> currentSamples = new List<float>();

		private int lastWriteTime = 0;

		public PCM2MIDI() {
			super(P);
		}

		@Override
		protected void init(IAudioRenderTarget target) throws RenderCommandException {
			super.init(target);
			midiRef.clear();
			for(MidiEvent e : getRefMidi()) {
				MidiMessage msg = e.getMessage();
				if(msg instanceof ShortMessage &&
						(msg.getMessage()[0] & 0xFF) != ShortMessage.NOTE_ON ||
						(msg.getMessage()[2] & 0xFF) == 0) continue;
				int msTime = (int) (e.getTick() / 1000L);
				while(midiRef.size() <= msTime)
					midiRef.add(null);
				List<MidiEvent> evts = midiRef.get(msTime);
				if(evts == null) {
					evts = new ArrayList<MidiEvent>();
					midiRef.set(msTime, evts);
				}
				evts.add(e);
			}
		}

		@Override
		protected void run(IAudioRenderTarget target) throws RenderCommandException {
			try {
				int msTimeLimit = (int) (target.getFrame().playOutTime * IScheduler.SEC2MS);
				for(;msTime <= msTimeLimit; msTime++) {
					if(msTime < midiRef.size()) {
						List<MidiEvent> evts = midiRef.get(msTime);
						if(evts != null) {
							for(MidiEvent e : evts){
								currentMidiRef.add(e.getMessage().getMessage()[1]);  //Liste der aktuell aktiven Midi-Werte
							}
						}

						currentSamples.add(target.getFrame().samples);
					}

					if(msTime - lastWriteTime > 50){
						File.Write(String.join(",", currentSamples) + "---" + String.join(",", currentMidiRef) + "\n");  //Pseudocode!
						currentSamples = new List<float>();
						currentMidiRef = new HastSet<int>();

						lastWriteTime = msTime;
					}
				}
			} catch(Throwable t) {
				throw new RenderCommandException(t);
			}
		}
	}

}
