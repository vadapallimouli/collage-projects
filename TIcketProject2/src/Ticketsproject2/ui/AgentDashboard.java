package Ticketsproject2.ui;

import java.awt.EventQueue;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.table.DefaultTableModel;

import Ticketsproject2.database.DBConnection;

public class AgentDashboard extends JFrame {

	private static final long serialVersionUID = 1L;

	private JTextField textField;
	private JTextField textField_1;
	private JTable table;
	private JComboBox<String> comboBox_1;

	private JLabel lblNewLabel_5;
	private JLabel lblNewLabel_6;
	private JLabel lblNewLabel_7;

	private String agentUsername = "agent1";

	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
			try {
				AgentDashboard frame = new AgentDashboard("agent1");
				frame.setVisible(true);
			} catch (Exception e) {
				e.printStackTrace();
			}
		});
	}

	public AgentDashboard() {
		this("agent1");
	}

	public AgentDashboard(String agentUsername) {
		this.agentUsername = agentUsername;

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 1000, 700);
		getContentPane().setLayout(null);

		JLabel lblNewLabel = new JLabel("AGENT DASHBOARD");
		lblNewLabel.setBounds(20, 15, 220, 30);
		getContentPane().add(lblNewLabel);

		JLabel lblAgentName = new JLabel("Logged in as: " + this.agentUsername);
		lblAgentName.setBounds(240, 15, 200, 30);
		getContentPane().add(lblAgentName);

		JButton btnNewButton = new JButton("Logout");
		btnNewButton.setBounds(860, 15, 100, 30);
		getContentPane().add(btnNewButton);

		JLabel lblNewLabel_1 = new JLabel("Open Tickets");
		lblNewLabel_1.setBounds(50, 70, 100, 20);
		getContentPane().add(lblNewLabel_1);

		JLabel lblNewLabel_2 = new JLabel("In Progress");
		lblNewLabel_2.setBounds(300, 70, 100, 20);
		getContentPane().add(lblNewLabel_2);

		JLabel lblNewLabel_3 = new JLabel("Total Active");
		lblNewLabel_3.setBounds(550, 70, 100, 20);
		getContentPane().add(lblNewLabel_3);

		lblNewLabel_5 = new JLabel("0");
		lblNewLabel_5.setBounds(85, 100, 40, 20);
		getContentPane().add(lblNewLabel_5);

		lblNewLabel_6 = new JLabel("0");
		lblNewLabel_6.setBounds(335, 100, 40, 20);
		getContentPane().add(lblNewLabel_6);

		lblNewLabel_7 = new JLabel("0");
		lblNewLabel_7.setBounds(585, 100, 40, 20);
		getContentPane().add(lblNewLabel_7);

		JLabel lblNewLabel_10 = new JLabel("Selected ID:");
		lblNewLabel_10.setBounds(20, 150, 90, 20);
		getContentPane().add(lblNewLabel_10);

		textField_1 = new JTextField();
		textField_1.setBounds(110, 147, 120, 28);
		textField_1.setColumns(10);
		textField_1.setEditable(false);
		getContentPane().add(textField_1);

		JLabel lblNewLabel_11 = new JLabel("Status:");
		lblNewLabel_11.setBounds(280, 150, 60, 20);
		getContentPane().add(lblNewLabel_11);

		comboBox_1 = new JComboBox<>();
		comboBox_1.setModel(new DefaultComboBoxModel<>(new String[] {
				"In Progress", "Closed"
		}));
		comboBox_1.setBounds(340, 147, 150, 28);
		getContentPane().add(comboBox_1);

		JLabel lblSearch = new JLabel("Search by ID:");
		lblSearch.setBounds(20, 215, 90, 20);
		getContentPane().add(lblSearch);

		textField = new JTextField();
		textField.setBounds(110, 210, 180, 30);
		textField.setColumns(10);
		getContentPane().add(textField);

		JButton btnNewButton_1 = new JButton("Search");
		btnNewButton_1.setBounds(320, 210, 100, 30);
		getContentPane().add(btnNewButton_1);

		JButton btnNewButton_2 = new JButton("Refresh");
		btnNewButton_2.setBounds(440, 210, 100, 30);
		getContentPane().add(btnNewButton_2);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(20, 270, 940, 300);
		getContentPane().add(scrollPane);

		table = new JTable();
		table.setModel(new DefaultTableModel(
			new Object[][] {
			},
			new String[] {
				"ID", "Customer", "Issue", "Priority", "Assigned Agent", "Status"
			}
		) {
			private static final long serialVersionUID = 1L;

			@Override
			public boolean isCellEditable(int row, int column) {
				return false;
			}
		});
		scrollPane.setViewportView(table);

		JButton btnNewButton_3 = new JButton("Update Status");
		btnNewButton_3.setBounds(20, 590, 140, 35);
		getContentPane().add(btnNewButton_3);

		btnNewButton.addActionListener(e -> dispose());
		btnNewButton_1.addActionListener(e -> searchTicketById());
		btnNewButton_2.addActionListener(e -> refreshDashboard());
		btnNewButton_3.addActionListener(e -> updateTicketStatus());

		table.getSelectionModel().addListSelectionListener(e -> {
			if (!e.getValueIsAdjusting()) {
				fillFieldsFromTable();
			}
		});

		loadVisibleTickets();
		loadStats();
	}

	private void loadVisibleTickets() {
		DefaultTableModel model = (DefaultTableModel) table.getModel();
		model.setRowCount(0);

		String sql = "SELECT id, customerName, issue, priority, assignedAgent, status "
				+ "FROM tickets WHERE assignedAgent = ? AND (status = 'Open' OR status = 'In Progress') "
				+ "ORDER BY id DESC";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setString(1, agentUsername);

			try (ResultSet rs = pst.executeQuery()) {
				while (rs.next()) {
					model.addRow(new Object[] {
						rs.getInt("id"),
						rs.getString("customerName"),
						rs.getString("issue"),
						rs.getString("priority"),
						rs.getString("assignedAgent"),
						rs.getString("status")
					});
				}
			}
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Error loading tickets: " + e.getMessage());
		}
	}

	private void searchTicketById() {
		String idText = textField.getText().trim();

		if (idText.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Enter ticket ID");
			return;
		}

		DefaultTableModel model = (DefaultTableModel) table.getModel();
		model.setRowCount(0);

		String sql = "SELECT id, customerName, issue, priority, assignedAgent, status "
				+ "FROM tickets WHERE id = ? AND assignedAgent = ? "
				+ "AND (status = 'Open' OR status = 'In Progress')";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setInt(1, Integer.parseInt(idText));
			pst.setString(2, agentUsername);

			try (ResultSet rs = pst.executeQuery()) {
				if (rs.next()) {
					model.addRow(new Object[] {
						rs.getInt("id"),
						rs.getString("customerName"),
						rs.getString("issue"),
						rs.getString("priority"),
						rs.getString("assignedAgent"),
						rs.getString("status")
					});

					textField_1.setText(String.valueOf(rs.getInt("id")));
					comboBox_1.setSelectedItem(rs.getString("status").equals("Open") ? "In Progress" : rs.getString("status"));
				} else {
					JOptionPane.showMessageDialog(this, "Ticket not found for this agent");
				}
			}
		} catch (NumberFormatException e) {
			JOptionPane.showMessageDialog(this, "Ticket ID must be a number");
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Search error: " + e.getMessage());
		}
	}

	private void fillFieldsFromTable() {
		int row = table.getSelectedRow();

		if (row >= 0) {
			textField_1.setText(table.getValueAt(row, 0).toString());
			String currentStatus = table.getValueAt(row, 5).toString();

			if ("Open".equals(currentStatus)) {
				comboBox_1.setSelectedItem("In Progress");
			} else {
				comboBox_1.setSelectedItem(currentStatus);
			}
		}
	}

	private void updateTicketStatus() {
		String selectedId = textField_1.getText().trim();

		if (selectedId.isEmpty()) {
			JOptionPane.showMessageDialog(this, "Select a ticket first");
			return;
		}

		String sql = "UPDATE tickets SET status = ? WHERE id = ? AND assignedAgent = ? "
				+ "AND (status = 'Open' OR status = 'In Progress')";

		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setString(1, comboBox_1.getSelectedItem().toString());
			pst.setInt(2, Integer.parseInt(selectedId));
			pst.setString(3, agentUsername);

			int updated = pst.executeUpdate();

			if (updated > 0) {
				JOptionPane.showMessageDialog(this, "Ticket updated successfully");
				refreshDashboard();
			} else {
				JOptionPane.showMessageDialog(this, "Only open or in progress tickets can be updated");
			}
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Update error: " + e.getMessage());
		}
	}

	private void refreshDashboard() {
		textField.setText("");
		textField_1.setText("");
		comboBox_1.setSelectedItem("In Progress");
		loadVisibleTickets();
		loadStats();
	}

	private void loadStats() {
		lblNewLabel_5.setText(String.valueOf(getCount(
				"SELECT COUNT(*) FROM tickets WHERE assignedAgent = ? AND status = 'Open'")));

		lblNewLabel_6.setText(String.valueOf(getCount(
				"SELECT COUNT(*) FROM tickets WHERE assignedAgent = ? AND status = 'In Progress'")));

		lblNewLabel_7.setText(String.valueOf(getCount(
				"SELECT COUNT(*) FROM tickets WHERE assignedAgent = ? AND (status = 'Open' OR status = 'In Progress')")));
	}

	private int getCount(String sql) {
		try (Connection con = DBConnection.getConnection();
			 PreparedStatement pst = con.prepareStatement(sql)) {

			pst.setString(1, agentUsername);

			try (ResultSet rs = pst.executeQuery()) {
				if (rs.next()) {
					return rs.getInt(1);
				}
			}
		} catch (SQLException e) {
			JOptionPane.showMessageDialog(this, "Stats error: " + e.getMessage());
		}
		return 0;
	}
}
