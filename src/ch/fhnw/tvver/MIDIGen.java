package ch.fhnw.tvver;

import java.io.File;
import java.io.IOException;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;

public class MIDIGen {

	public static void main(String[] args) throws InvalidMidiDataException, IOException {
		Sequence sequence = new Sequence(Sequence.PPQ, 2);

		Track	track = sequence.createTrack();

		long tick = 0;
		for(int i = Integer.parseInt(args[1]); i <= Integer.parseInt(args[2]); i++) {
			track.add(new MidiEvent(new ShortMessage(ShortMessage.NOTE_ON, 0, i, 64), tick));
			tick++;
			track.add(new MidiEvent(new ShortMessage(ShortMessage.NOTE_OFF, 0, i, 64), tick));
			tick++;
		}

		MidiSystem.write(sequence, 0, new File(args[0]));
	}

}
