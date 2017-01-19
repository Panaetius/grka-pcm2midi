package ch.fhnw.tvver;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.ShortMessage;

import ch.fhnw.ether.media.AbstractRenderCommand;
import ch.fhnw.ether.media.IScheduler;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.media.RenderProgram;
import ch.fhnw.ether.midi.IMidiRenderTarget;
import ch.fhnw.ether.midi.MidiFrame;
import ch.fhnw.ether.midi.NullMidiTarget;
import ch.fhnw.ether.midi.URLMidiSource;

public class MIDIKeyHistogram {
	static int[] keys = new int[128];

	private static final String[] EXCLUDE = {"scale.mid"};

	public static void main(String[] args) throws MalformedURLException, InvalidMidiDataException, IOException, MidiUnavailableException, RenderCommandException {
		scan(new File(args[0]));

		for(int i = 0; i < keys.length; i++)
			System.out.println(i + "\t" + keys[i]);
	}

	private static void scan(File file) throws MalformedURLException, InvalidMidiDataException, IOException, MidiUnavailableException, RenderCommandException {
		if(file.isDirectory()) {
			for(File f : file.listFiles())
				scan(f);
		} else if(file.isFile() && file.getName().endsWith(".mid")) {
			for(String excl : EXCLUDE)
				if(file.getName().equals(excl))
					return;
			URLMidiSource src = new URLMidiSource(file.toURI().toURL(), 1);
			RenderProgram<IMidiRenderTarget> program = new RenderProgram<IMidiRenderTarget>(src, new AbstractRenderCommand<IMidiRenderTarget>() {
				@Override
				protected void run(IMidiRenderTarget target) throws RenderCommandException {
					MidiFrame frame = target.getFrame();
					for(MidiMessage m : frame.messages) {
						if(m instanceof ShortMessage) {
							ShortMessage sm = (ShortMessage)m;
							if(sm.getCommand() == ShortMessage.NOTE_ON) {
								keys[sm.getData1()]++;
							}
						}
					}
				}
			});
			NullMidiTarget out = new NullMidiTarget(false);
			out.useProgram(program);
			out.start();
			out.sleepUntil(IScheduler.NOT_RENDERING);
			out.stop();
		}
	}
}
