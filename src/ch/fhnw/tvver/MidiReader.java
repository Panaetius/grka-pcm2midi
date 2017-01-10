package ch.fhnw.tvver;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiFileFormat;
import javax.sound.midi.Sequence;
import javax.sound.midi.spi.MidiFileReader;

public class MidiReader extends MidiFileReader {

	@Override
	public MidiFileFormat getMidiFileFormat(InputStream stream) throws InvalidMidiDataException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MidiFileFormat getMidiFileFormat(URL url) throws InvalidMidiDataException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MidiFileFormat getMidiFileFormat(File file) throws InvalidMidiDataException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Sequence getSequence(InputStream stream) throws InvalidMidiDataException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Sequence getSequence(URL url) throws InvalidMidiDataException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Sequence getSequence(File file) throws InvalidMidiDataException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

}
